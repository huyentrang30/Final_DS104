from pyspark.sql import DataFrame

# {{{ Column classification
IDENTIFIER = ['MaTin']

CONTINUOUS_COLUMNS = [
    'TongGia',
    'Gia/m2',
    'DienTich',
    'DienTichDat',
    'ChieuDai',
    'ChieuRong',
    'ChieuSau',
    'DuongVao',
    'NamXayDung'
]

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
# }}}

# Count null value
def countNull(df: DataFrame) -> DataFrame:
    import plotly.express as px
    from pyspark.sql.functions import col, when, count

    df_count = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
    pdfDF = df_count.toPandas().T
    pdfDF.columns = ['No. null value']
    px.bar(pdfDF, text_auto=True, title="Count number of null value each features").show()
    
    return df_count

# Fill, replace empty value with None/null
def fillEmptyValue(df: DataFrame) -> DataFrame:
    from pyspark.sql.functions import col, when

    for c in df.columns:
        if c not in STRUCTURED_COLUMNS:
            df = df.withColumn(c, when(col(c)=='--', None).otherwise(col(c)).alias(c))\
                .withColumn(c, when(col(c)=='', None).otherwise(col(c)).alias(c))

    return df

# {{{ Remove unit, convert string to castable
def rmU_TongGia(TongGia):
    if TongGia is not None:
        gia = TongGia.split(' ')
        if gia[1] == 'tỷ':
            return float(gia[0])*1000.0
        elif gia[1] == 'triệu':
            return float(gia[0])
        elif gia[1] == 'nghìn':
            return round(float(gia[0])*0.001, 3)
        else:
            return TongGia
    else: 
        return None

def rmU_GiaM2(GiaM2):
    if GiaM2 is not None:
        gia = GiaM2.split(' ')
        if gia[1] == 'tỷ/m²':
            return float(gia[0])*1000.0
        elif gia[1] == 'triệu/m²':
            return float(gia[0])
        elif gia[1] == 'nghìn/m²':
            return round(float(gia[0])*0.001, 3)
    else:
        return None

def removeUnitString(df: DataFrame) -> DataFrame:
    from pyspark.sql.functions import udf

    udf_TongGia= udf(rmU_TongGia)
    udf_GiaM2= udf(rmU_GiaM2)
    udf_DienTich = udf(lambda S: S[:-3] if S is not None else None)
    udf_Dist = udf(lambda D: D[:-2] if D is not None else None)

    for col in df.columns:
        if col == 'TongGia':
            df = df.withColumn(col, udf_TongGia(col))
        if col == 'Gia/m2':
            df = df.withColumn(col, udf_GiaM2(col))
        if col in ['DienTich','DienTichDat']:
            df = df.withColumn(col, udf_DienTich(col))
        if col in ['ChieuDai','ChieuRong','ChieuSau','DuongVao']:
            df = df.withColumn(col, udf_Dist(col))

    return df
# }}}

# {{{ Missing value imputed
def fillNullValue(df: DataFrame) -> DataFrame:
    from pyspark.sql.functions import coalesce, col, avg, array

    if ('DienTich' in df.columns and 'DienTichDat' in df.columns):
        df = df.withColumn('DienTich', coalesce('DienTich', 'DienTichDat'))

    for column in df.columns: 

        if column in ['NamXayDung']:
            mode = df.select(column).where(col(column).isNotNull())\
                .groupby(column).count()\
                .orderBy("count", ascending=False).first()[0]
            df = df.fillna(value=mode, subset=[column])

        if column in ['ChieuDai', 'ChieuRong', 'DuongVao','DienTich']:
            mean = df.agg(avg(column)).first()[0]
            df = df.fillna(value=str(mean), subset=[column])

        if column in ['PhongNgu', 'PhongTam']:
            df = df.fillna(value='0', subset=[column])

        if column in ['SoTang']:
            df = df.fillna(value='1', subset=[column])

        if column in CATEGORICAL_COLUMNS:
            df = df.fillna(value='Unknowns', subset=[column])

        if column in STRUCTURED_COLUMNS:
            df = df.withColumn(column, coalesce(column, array()))


    return df
# }}}

# {{{ Removing useless records, rare features and type casting
def dropData(df: DataFrame, isTrain=True) -> DataFrame:
    if isTrain:
        df = df.dropna(subset=['MaTin','NgayDangBan','TongGia'], how='any')
    else:
        df = df.dropna(subset=['MaTin','NgayDangBan'], how='any')

    df = df.drop(*['DienTichDat','ChieuSau'])
    return df

def typeCasting(df: DataFrame) -> DataFrame:
    from pyspark.sql.functions import col
    
    int_columns = ['NamXayDung']
    float_columns = [
        c for c in CONTINUOUS_COLUMNS
        if c not in int_columns
        and c not in ['ChieuSau','DienTichDat']]
    for c in df.columns:
        if c in int_columns:
            df = df.withColumn(c, col(c).cast('int'))
        elif c in float_columns:
            df = df.withColumn(c, col(c).cast('float'))
    return df
# }}}

def from_pd_to_spark(df):
    from pyspark.sql.functions import udf
    from pyspark.sql.types import ArrayType, StringType
    from utils import convert_to_list

    convert_to_list_udf = udf(lambda x: convert_to_list(x), ArrayType(StringType()))
    
    for col in STRUCTURED_COLUMNS:
        if col in df.columns:
            df = df.withColumn(col, convert_to_list_udf(df[col]))

    return df

def cleanRawData(df: DataFrame, isTrain=True) -> DataFrame:
    df1 = fillEmptyValue(df)
    df2 = removeUnitString(df1)
    df3 = fillNullValue(df2)
    df4 = dropData(df3, isTrain)
    df5 = typeCasting(df4)
    return df5