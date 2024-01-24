from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoderModel
from typing import Tuple, List

from utils import *

STRUCTURED_COLUMNS = [
    'TienIchToaNha',
    'TienIchLanCan',
    'NoiThat,TienNghi',
    'HangXom',
    'AnNinh',
    'DuongVaoBds',
    'TienIchGanDat'
]

CATEGORICAL_COLUMNS = [
    'Id_NguoiDangban',
    'LoaiBDS',
    'Tinh',
    'Xa',
    'Huyen',
    'Huong',
    'PhongNgu',
    'PhongTam',
    'GiayToPhapLy',
    'SoTang',
    'ViTri',
    'HienTrangNha',
    'TrangThaiSuDung',
    'HuongBanCong',
    'KetCau'
]

class OHE():
    '''
    Sử dụng One Hot Encoder để transfom dữ liệu từ Array<value> -> Vector[value].
    * categories : {'auto', array<>}, default='auto'.
        - Tự động xác định danh sách giá trị phân loại hoặc sử dungk danh sách giá trị
        phân loại được cấp từ array 
    * dropInput : bool, defauly=False: Bỏ cột thuộc tính đầu vào)
    '''
    def __init__(self, categories='auto', dropInput=False):
        self.categories = categories
        self.dropInput = dropInput

    def transform(self, df, feature):
        from pyspark.sql.functions import monotonically_increasing_id, regexp_replace, explode_outer, col, lit

        self.feature = feature
        df = df.withColumn("_idx", monotonically_increasing_id() )

        if self.categories == 'auto':
            _idPrefix = '{0}_{1}'.format('_idx', feature)

            explode_df = df\
                .withColumn(self.feature, explode_outer(self.feature))\
                .withColumn(feature, regexp_replace(feature, r'\.',''))
            crosstab_df = explode_df.crosstab('_idx', self.feature).drop('null')

            cats_df = crosstab_df

            current_categories = cats_df.columns
            current_categories.sort()
            cats_df = cats_df.select(current_categories)
            current_categories.remove(_idPrefix)
            self.categories = current_categories

            categories_order = [col(col_name).alias("{0}_".format(feature) + col_name) 
                                      for col_name in cats_df.columns if col_name != _idPrefix]
            cats_df = cats_df.select(*categories_order + [col(_idPrefix)])
            self.categories_prefix = cats_df.columns
            self.categories_prefix.remove(_idPrefix)

            df = df.join(cats_df, cats_df[_idPrefix]==df['_idx']).drop(*['_idx',_idPrefix])

        else:
            _idPrefix = '{0}_{1}'.format('_idx', feature)

            explode_df = df\
                .withColumn(self.feature, explode_outer(self.feature))\
                .withColumn(feature, regexp_replace(feature, r'\.',''))
            crosstab_df = explode_df.crosstab('_idx', self.feature).drop('null')

            cats_df = crosstab_df

            new_cats = [c for c in cats_df.columns if c not in self.categories+[_idPrefix]]
            miss_cats = [c for c in self.categories if c not in cats_df.columns]

            cats_df = cats_df.drop(*new_cats)

            for c in miss_cats:
                cats_df = cats_df.withColumn(c, lit(0))

            self.categories.sort()
            cats_df = cats_df.select(self.categories + [_idPrefix])

            categories_order = [col(col_name).alias("{0}_".format(feature) + col_name) 
                                      for col_name in cats_df.columns if col_name != _idPrefix]

            cats_df = cats_df.select(*categories_order + [col(_idPrefix)])
            self.categories_prefix = cats_df.columns
            self.categories_prefix.remove(_idPrefix)

            df = df.join(cats_df, cats_df[_idPrefix]==df['_idx']).drop(*['_idx',_idPrefix])

        if self.dropInput:
            return df.drop(feature)
        else:
            return df


def OHEtransform(
    df: DataFrame, isPredict=False, keepInput=False, keepOutput=False,
    vectorizeEach=True, vectorize=True, outputCol='features_ohe'
    ) -> Tuple[DataFrame, List[PipelineModel]]:

    df_trans = df
    categories = 'auto'
    ohe_models = []
    vec_names = []
    save_cats = {}

    if vectorize:
        vectorizeEach=True

    if isPredict:
        save_cats = load_ohe_categories()

    for feature in df.columns:
        if feature in STRUCTURED_COLUMNS:

            if isPredict:
                categories = save_cats[feature]

            encoder = OHE(categories=categories)
            df_trans = encoder.transform(df_trans, feature)
            ohe_models.append(encoder)

            if vectorizeEach:
                vec_name_prefix = '{0}_ohe'.format(feature)
                vec_names.append(vec_name_prefix)
                assembler = VectorAssembler(inputCols = encoder.categories_prefix,
                                            outputCol = vec_name_prefix)
                df_trans = assembler.transform(df_trans)

                if keepOutput:
                    pass
                else:
                    df_trans = df_trans.drop(*encoder.categories_prefix)
            else:
                pass

            if keepInput:
                pass
            else:
                df_trans = df_trans.drop(feature)
    
    if vectorize:
        assembler = VectorAssembler(
            inputCols=vec_names,
            outputCol=outputCol)
        df_trans = assembler.transform(df_trans)
        df_trans = df_trans.drop(*vec_names)

    return df_trans, ohe_models


def binningDistribute(df: DataFrame, keepInput=False) -> DataFrame:
    from pyspark.ml.feature import Bucketizer

    df_distribution = df.groupBy('Id_NguoiDangban').count()#.sort(f.col('count').desc())
    bucketizer = Bucketizer(splits=[0,2,5,8,16,float("Inf")], inputCol="count", outputCol="Id_NguoiDangban_level")
    df_distribution_level = bucketizer.setHandleInvalid("keep").transform(df_distribution)

    if keepInput:
        return df.join(df_distribution_level,'Id_NguoiDangban').drop(*['count'])
    else:
        return df.join(df_distribution_level,'Id_NguoiDangban').drop(*['count','Id_NguoiDangban'])


def getDummy(
    df, keepInput=False, keepOutput=False, vectorize=True, outputCol='features_idx',
    isPredict = False, models_stringIndex = None):

    idx_columns = [c for c in CATEGORICAL_COLUMNS if not c in ['Tinh','Huyen','Xa','Id_NguoiDangban_level']]
    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_idx".format(c), handleInvalid="skip")
                 for c in df.columns if c in idx_columns]

    pipeline1 = Pipeline(stages=indexers)
    if isPredict:
        pass
    else:
        models_stringIndex = pipeline1.fit(df)

    data = models_stringIndex.transform(df)

    if keepInput:
        pass
    else:
        data = data.drop(*[c for c in df.columns if c in idx_columns])

    if vectorize:
        assembler = VectorAssembler(
            inputCols=[indexer.getOutputCol() for indexer in indexers],
            outputCol=outputCol)
        data = assembler.transform(data)

        if keepOutput:
            pass
        else:
            data = data.drop(*['{0}_idx'.format(c) for c in df.columns if c in idx_columns])
    else:
        pass
        
    return data, models_stringIndex


def getEncodedDummy(df, keepInput = False, isPredict = False, encoder_m = None):
    import re
    from pyspark.ml.feature import OneHotEncoder
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    pat = re.compile(r"^.*_idx.*$")
    idxList = [i for i in df.columns if pat.match(i)]
    oheList = [sub.replace('idx', 'ohe') for sub in idxList]

    transform_empty = udf(lambda s: "NA" if s == "" else s, StringType())
    for col in idxList:
        df = df.withColumn(col, transform_empty(col))
        df = df.withColumn(col, df[col].cast('double'))

    encoder = OneHotEncoder(inputCols=idxList, outputCols=oheList)
    if isPredict and encoder_m:
        pass
    else:
        encoder_m = encoder.fit(df)
    encoded = encoder_m.transform(df)
    
    if keepInput:
        return encoded, encoder_m
    else:
        return encoded.drop(*idxList), encoder_m


def getAdministrative(df: DataFrame, vectorize=True, keepInput=False, keepOutput=False, outputCol='features_adm') -> DataFrame:
    from pyspark.sql.functions import col, when, lower
    from utils import initialize_spark
    spark, _ = initialize_spark()
    Idxs = []
    for feature in df.columns:
        if feature == 'Tinh':
            import pandas as pd
            pd.DataFrame.iteritems = pd.DataFrame.items
            provinces_tier = pd.read_csv('data/provinces_tier.csv')
            provinces_tier = spark.createDataFrame(provinces_tier)
            df = df.join(provinces_tier, ['Tinh'], how='left')

            df = df.withColumn('Tinh_idx',
                when(col('PhanLoaiTinh').contains('Đặc biệt'), 4)\
                .when(col('PhanLoaiTinh').contains('III'), 3)\
                .when(col('PhanLoaiTinh').contains('II'), 2)\
                .when(col('PhanLoaiTinh').contains('I'), 1)
                .otherwise(1))
            Idxs.append('Tinh_idx')

        elif feature == 'Huyen':
            df = df.withColumn('Huyen_idx',
                when(lower(col('Huyen')).contains('tp'),3)\
                .when(lower(col('Huyen')).contains('quận'),3)\
                .when(lower(col('Huyen')).contains('thị xã'),2)\
                .when(lower(col('Huyen')).contains('huyện'),1)
                .otherwise(1))
            Idxs.append('Huyen_idx')

        elif feature == 'Xa':
            df = df.withColumn('Xa_idx',
                when(lower(col('Xa')).contains('phường'), 3)\
                .when(lower(col('Xa')).contains('thị trấn'), 2)\
                .when(lower(col('Xa')).contains('xã'), 1)
                .otherwise(1))
            Idxs.append('Xa_idx')
    
    if vectorize:
        assembler = VectorAssembler(
            inputCols=Idxs,
            outputCol=outputCol)
        df = assembler.transform(df)
        if keepOutput:
            pass
        else:
            df = df.drop(*Idxs+['PhanLoaiTinh'])
    else:
        pass

    if keepInput:
        pass
    else:
        df = df.drop(*[c for c in df.columns if c in ['Tinh','Huyen','Xa']]+['PhanLoaiTinh'])

    return df


def featureExtraction(df: DataFrame, string_idx=None, enc_m=None) -> DataFrame:
    
    df1 = binningDistribute(df)
    df2 = getAdministrative(df1, vectorize=False)
    df3, stringIndexs = getDummy(df2, isPredict=True, vectorize=False, models_stringIndex=string_idx)
    df4, encodes = OHEtransform(df3, isPredict=True, vectorize=False)
    df5, en_m = getEncodedDummy(df4, isPredict=True, encoder_m=enc_m)

    # if vectorize:
    #     assembler = VectorAssembler(
    #         inputCols=['features_adm','features_idx','features_ohe'],
    #         outputCol=outputCol)
    #     data = assembler.transform(df4)
    # else:
    #     data = df4

    features = df5.columns
    features = [ele for ele in features if ele not in ['DiaChi','Gia/m2', 'MaTin','NgayDangBan', 'TongGia', 'NguoiDangban']]
    assembler = VectorAssembler(inputCols = features, outputCol="features")

    assembled_df = assembler.transform(df5)

    return assembled_df