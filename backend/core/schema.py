from backend.core.db import execute


def create_feature_history_table():

    execute("""

    CREATE TABLE IF NOT EXISTS feature_history (

        symbol TEXT NOT NULL,
        date TEXT NOT NULL,

        alma REAL,
        dema REAL,

        ema_20 REAL,
        ema_50 REAL,
        ema_100 REAL,
        ema_200 REAL,

        hma_20 REAL,
        kama REAL,
        lsma REAL,
        mcginley_dynamic REAL,

        sma_20 REAL,
        sma_50 REAL,
        sma_100 REAL,
        sma_200 REAL,

        smma REAL,

        t3 REAL,
        tema REAL,
        trima REAL,

        vidya REAL,

        vwma_20 REAL,

        wma_20 REAL,

        zlema REAL,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        PRIMARY KEY (symbol, date)

    )

    """)


def create_all_tables():

    create_feature_history_table()


if __name__ == "__main__":

    create_all_tables()

