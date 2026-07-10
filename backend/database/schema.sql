
CREATE TABLE IF NOT EXISTS feature_history (

    symbol TEXT NOT NULL,

    date TEXT NOT NULL,

    ALMA_9 REAL,
    DEMA_20 REAL,

    EMA_20 REAL,
    EMA_50 REAL,
    EMA_100 REAL,
    EMA_200 REAL,

    HMA_20 REAL,
    KAMA_10 REAL,
    LSMA_25 REAL,
    MCGINLEY_14 REAL,

    SMA_20 REAL,
    SMA_50 REAL,
    SMA_100 REAL,
    SMA_200 REAL,

    SMMA_20 REAL,

    T3_5 REAL,
    TEMA_20 REAL,
    TRIMA_20 REAL,

    VIDYA_9 REAL,

    VWMA_20 REAL,

    WMA_20 REAL,

    ZLEMA_20 REAL,

    PRIMARY KEY (
        symbol,
        date
    )

);

