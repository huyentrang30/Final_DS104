import sklearn
import pandas as pd

from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorSlicer

def prediction(model, data, labelCol, predictionCol='prediction') -> DataFrame:
    import math
    from pyspark.sql.functions import col, lit, pow

    prediction = model.transform(data)\
        .withColumn(labelCol, pow(lit(math.e), col(labelCol)))\
        .withColumn(predictionCol, pow(lit(math.e), col(predictionCol)))

    prediction.select(labelCol,predictionCol).show(10)

    return prediction


def evaluation(prediction, labelCol, predictionCol='prediction'):
    from pyspark.ml.evaluation import RegressionEvaluator

    evaluator = RegressionEvaluator(labelCol)

    rmse_test = evaluator.evaluate(prediction)
    print('RMSE: %f triệu VND' %rmse_test)

    mae_test = evaluator.evaluate(prediction, {evaluator.metricName: "mae"})
    print('MAE: %f triệu VND' %mae_test)

    y_true = prediction.select(labelCol).toPandas()
    y_pred = prediction.select(predictionCol).toPandas()

    r2_score = sklearn.metrics.r2_score(y_true, y_pred)
    print('R2: %f' %(r2_score))


def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))


def FeatureImpSelector(dataset, estimator, selectorType='topFeatures', nFeatures=20, threshold=0.01,
                       inputCol='features', outputCol='features_slice'):
    mod = estimator.fit(dataset)
    dataset2 = mod.transform(dataset)
    varlist = ExtractFeatureImp(mod.featureImportances, dataset2, inputCol)

    if (selectorType == "topFeatures"):
        varidx = [x for x in varlist['idx'][0:nFeatures]]
    elif (selectorType == "threshold"):
        varidx = [x for x in varlist[varlist['score'] > threshold]['idx']]

    return VectorSlicer(inputCol = inputCol, outputCol = outputCol, indices = varidx)