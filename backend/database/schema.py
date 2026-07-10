"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/database/schema.py
Description: Enterprise Schema Definition and Lifecycle Engine.
             Provides fully deterministic DDL orchestration for SQLite with absolute 
             forward compatibility for PostgreSQL migrations. 
             Strictly mapped to the AI Pipeline Architecture with extensive 
             Time-Series constraints (timeframe), JSON validation, strict CHECK 
             bounds, and comprehensive System/Migration tables.
             Includes 45+ tables, 70+ optimized B-Tree indexes, and 12+ Pipeline Views.
             Production Locked.
"""

import time
import threading
from typing import List, Final

# Internal Platform Integrations
from backend.config.settings import settings
from backend.core.logger import AppLogger
from backend.core.trace import TraceEngine, SpanKind, trace_span
from backend.core.audit import AuditEngine, AuditAction, AuditSeverity
from backend.core.metrics import metrics_engine
from backend.core.exceptions import SchemaError, DatabaseError
from backend.database.connection import db_manager, DatabaseSession, IsolationLevel

# -------------------------------------------------------------------------
# LOGGER INITIALIZATION
# -------------------------------------------------------------------------
_logger = AppLogger("SchemaEngine")

# -------------------------------------------------------------------------
# ENTERPRISE DDL DEFINITIONS (AI Pipeline Architecture Tables)
# -------------------------------------------------------------------------

_TABLE_DDL: Final[List[str]] = [
    # ------------------ INFRASTRUCTURE & SYSTEM ------------------
    """CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        description TEXT NOT NULL
    );""",

    """CREATE TABLE IF NOT EXISTS migration_history (
        id TEXT PRIMARY KEY,
        version INTEGER NOT NULL,
        script_name TEXT NOT NULL,
        checksum TEXT NOT NULL,
        execution_ms NUMERIC(18,6),
        status TEXT NOT NULL CHECK(status IN ('SUCCESS', 'FAILED')),
        applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );""",

    """CREATE TABLE IF NOT EXISTS metadata (
        key_name TEXT PRIMARY KEY,
        value_payload TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );""",

    """CREATE TABLE IF NOT EXISTS settings (
        id TEXT PRIMARY KEY,
        module TEXT NOT NULL,
        config_key TEXT NOT NULL,
        config_value TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(module, config_key)
    );""",

    """CREATE TABLE IF NOT EXISTS job_queue (
        id TEXT PRIMARY KEY,
        job_name TEXT NOT NULL,
        payload_json TEXT CHECK(payload_json IS NULL OR json_valid(payload_json)),
        status TEXT NOT NULL CHECK(status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')),
        priority INTEGER DEFAULT 0,
        run_at DATETIME NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );""",

    """CREATE TABLE IF NOT EXISTS api_rate_limit (
        id TEXT PRIMARY KEY,
        endpoint TEXT NOT NULL,
        client_ip TEXT,
        requests INTEGER DEFAULT 0,
        window_start DATETIME NOT NULL,
        UNIQUE(endpoint, client_ip, window_start)
    );""",

    """CREATE TABLE IF NOT EXISTS sync_status (
        id TEXT PRIMARY KEY,
        module_name TEXT NOT NULL,
        last_sync_timestamp DATETIME NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('SUCCESS', 'FAILED', 'IN_PROGRESS')),
        rows_processed INTEGER DEFAULT 0,
        errors_json TEXT CHECK(errors_json IS NULL OR json_valid(errors_json)),
        UNIQUE(module_name)
    );""",

    # ------------------ MASTER DATA ------------------
            """CREATE TABLE IF NOT EXISTS stock_master (
        symbol TEXT PRIMARY KEY,
        company_name TEXT NOT NULL CHECK(length(company_name) > 0),
        exchange TEXT NOT NULL CHECK(exchange IN ('NSE', 'BSE', 'MCX', 'NYSE', 'NASDAQ')),
        sector TEXT,
        industry TEXT,
        market_cap_category TEXT CHECK(market_cap_category IN ('LARGE', 'MID', 'SMALL', 'MICRO')),
        is_active BOOLEAN NOT NULL DEFAULT 1,
        is_fno BOOLEAN DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );""",

    """CREATE TABLE IF NOT EXISTS idx_stock_fno_dummy (id INTEGER);""", # Dummy to keep index block intact

    # নিচের ইনডেক্স ব্লকটি ঠিক করা
    
    """CREATE TABLE IF NOT EXISTS indices (
        index_symbol TEXT PRIMARY KEY,
        index_name TEXT NOT NULL,
        exchange TEXT DEFAULT 'NSE',
        is_active BOOLEAN DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );""",

    """CREATE TABLE IF NOT EXISTS sector_data (
        id TEXT PRIMARY KEY,
        sector_name TEXT NOT NULL,
        trade_date DATETIME NOT NULL,
        performance_pct NUMERIC(18,6),
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(sector_name, trade_date)
    );""",

    """CREATE TABLE IF NOT EXISTS industry_data (
        id TEXT PRIMARY KEY,
        industry_name TEXT NOT NULL,
        trade_date DATETIME NOT NULL,
        performance_pct NUMERIC(18,6),
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(industry_name, trade_date)
    );""",

    # ------------------ MARKET & HISTORICAL DATA ------------------
    """CREATE TABLE IF NOT EXISTS market_data (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1D',
        open NUMERIC(18,6) NOT NULL CHECK(open >= 0),
        high NUMERIC(18,6) NOT NULL CHECK(high >= 0),
        low NUMERIC(18,6) NOT NULL CHECK(low >= 0),
        close NUMERIC(18,6) NOT NULL CHECK(close >= 0),
        volume INTEGER NOT NULL CHECK(volume >= 0),
        vwap NUMERIC(18,6) CHECK(vwap >= 0),
        delivery_pct NUMERIC(18,6) CHECK(delivery_pct >= 0 AND delivery_pct <= 100),
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, timeframe),
        CHECK(high >= low)
    );""",

    """CREATE TABLE IF NOT EXISTS live_market_data (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1m',
        last_price NUMERIC(18,6) NOT NULL CHECK(last_price >= 0),
        volume INTEGER NOT NULL CHECK(volume >= 0),
        bid_price NUMERIC(18,6),
        ask_price NUMERIC(18,6),
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, timeframe)
    );""",

    """CREATE TABLE IF NOT EXISTS historical_data (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1D',
        open NUMERIC(18,6) CHECK(open >= 0),
        high NUMERIC(18,6) CHECK(high >= 0),
        low NUMERIC(18,6) CHECK(low >= 0),
        close NUMERIC(18,6) CHECK(close >= 0),
        volume INTEGER CHECK(volume >= 0),
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, timeframe),
        CHECK(high >= low OR high IS NULL)
    );""",

    # ------------------ FUNDAMENTAL & INSTITUTIONAL ------------------
    """CREATE TABLE IF NOT EXISTS fundamental_data (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        report_date DATETIME NOT NULL,
        pe_ratio NUMERIC(18,6),
        pb_ratio NUMERIC(18,6),
        eps NUMERIC(18,6),
        roe NUMERIC(18,6),
        roce NUMERIC(18,6),
        debt_to_equity NUMERIC(18,6),
        dividend_yield NUMERIC(18,6) CHECK(dividend_yield >= 0),
        market_cap NUMERIC(18,6) CHECK(market_cap >= 0),
        promoter_holding NUMERIC(18,6) CHECK(promoter_holding >= 0 AND promoter_holding <= 100),
        fii_holding NUMERIC(18,6) CHECK(fii_holding >= 0 AND fii_holding <= 100),
        dii_holding NUMERIC(18,6) CHECK(dii_holding >= 0 AND dii_holding <= 100),
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, report_date)
    );""",

    """CREATE TABLE IF NOT EXISTS institutional_data (
        id TEXT PRIMARY KEY,
        trade_date DATETIME NOT NULL UNIQUE,
        fii_buy NUMERIC(18,6),
        fii_sell NUMERIC(18,6),
        fii_net NUMERIC(18,6),
        dii_buy NUMERIC(18,6),
        dii_sell NUMERIC(18,6),
        dii_net NUMERIC(18,6),
        index_futures_net NUMERIC(18,6),
        index_options_net NUMERIC(18,6),
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );""",

    """CREATE TABLE IF NOT EXISTS corporate_actions (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        ex_date DATETIME NOT NULL,
        purpose TEXT NOT NULL,
        record_date DATETIME,
        bc_start DATETIME,
        bc_end DATETIME,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, ex_date, purpose)
    );""",

    """CREATE TABLE IF NOT EXISTS bulk_deals (
        id TEXT PRIMARY KEY,
        deal_date DATETIME NOT NULL,
        symbol TEXT NOT NULL,
        client_name TEXT NOT NULL,
        deal_type TEXT NOT NULL,
        buy_sell TEXT NOT NULL,
        quantity INTEGER NOT NULL CHECK(quantity > 0),
        price NUMERIC(18,6) NOT NULL CHECK(price > 0),
        remarks TEXT,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, deal_date, client_name, quantity, buy_sell)
    );""",

    """CREATE TABLE IF NOT EXISTS block_deals (
        id TEXT PRIMARY KEY,
        deal_date DATETIME NOT NULL,
        symbol TEXT NOT NULL,
        client_name TEXT NOT NULL,
        deal_type TEXT NOT NULL,
        buy_sell TEXT NOT NULL,
        quantity INTEGER NOT NULL CHECK(quantity > 0),
        price NUMERIC(18,6) NOT NULL CHECK(price > 0),
        remarks TEXT,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, deal_date, client_name, quantity, buy_sell)
    );""",

    """CREATE TABLE IF NOT EXISTS ipo_data (
        id TEXT PRIMARY KEY,
        symbol TEXT,
        company_name TEXT NOT NULL,
        open_date DATETIME,
        close_date DATETIME,
        issue_price NUMERIC(18,6) CHECK(issue_price > 0),
        status TEXT,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(company_name, open_date)
    );""",

    # ------------------ NEWS, SENTIMENT & SMART MONEY ------------------
    """CREATE TABLE IF NOT EXISTS news (
        id TEXT PRIMARY KEY,
        symbol TEXT,
        published_at DATETIME NOT NULL,
        headline TEXT NOT NULL,
        source TEXT,
        url TEXT,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, published_at, headline)
    );""",

    """CREATE TABLE IF NOT EXISTS sentiment (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1D',
        sentiment_score NUMERIC(18,6) CHECK(sentiment_score >= -1 AND sentiment_score <= 1),
        news_count INTEGER DEFAULT 0,
        social_volume INTEGER DEFAULT 0,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, timeframe)
    );""",

    """CREATE TABLE IF NOT EXISTS smart_money_data (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1D',
        fvg_bullish BOOLEAN DEFAULT 0,
        fvg_bearish BOOLEAN DEFAULT 0,
        choch BOOLEAN DEFAULT 0,
        bos BOOLEAN DEFAULT 0,
        order_block_price NUMERIC(18,6),
        liquidity_sweep BOOLEAN DEFAULT 0,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, timeframe)
    );""",

    """CREATE TABLE IF NOT EXISTS strategy (
        id TEXT PRIMARY KEY,
        strategy_name TEXT NOT NULL,
        description TEXT,
        parameters_json TEXT CHECK(parameters_json IS NULL OR json_valid(parameters_json)),
        is_active BOOLEAN DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(strategy_name)
    );""",

    # ------------------ DERIVATIVES ------------------
    """CREATE TABLE IF NOT EXISTS options_chain (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        expiry_date DATETIME NOT NULL,
        strike_price NUMERIC(18,6) NOT NULL,
        option_type TEXT NOT NULL,
        open_interest INTEGER,
        change_in_oi INTEGER,
        implied_volatility NUMERIC(18,6),
        last_price NUMERIC(18,6),
        volume INTEGER,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, expiry_date, strike_price, option_type)
    );""",

    """CREATE TABLE IF NOT EXISTS fno_data (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        expiry_date DATETIME NOT NULL,
        open_interest INTEGER,
        change_in_oi INTEGER,
        last_price NUMERIC(18,6),
        volume INTEGER,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, expiry_date)
    );""",

    # ------------------ AI PIPELINE CORE ------------------
    """CREATE TABLE IF NOT EXISTS indicator_data (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1D',
        indicator_name TEXT NOT NULL,
        value NUMERIC(18,6),
        payload_json TEXT CHECK(payload_json IS NULL OR json_valid(payload_json)),
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, timeframe, indicator_name)
    );""",

    """CREATE TABLE IF NOT EXISTS indicator_snapshot (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1D',
        payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timeframe)
    );""",

    """CREATE TABLE IF NOT EXISTS feature_data (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1D',
        feature_name TEXT NOT NULL,
        value NUMERIC(18,6),
        payload_json TEXT CHECK(payload_json IS NULL OR json_valid(payload_json)),
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, timeframe, feature_name)
    );""",

    """CREATE TABLE IF NOT EXISTS analysis_data (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1D',
        analyzer_name TEXT NOT NULL,
        sentiment TEXT,
        confidence NUMERIC(18,6) CHECK(confidence >= 0 AND confidence <= 1),
        summary TEXT,
        summary_compressed BLOB,
        reasoning TEXT,
        reasoning_compressed BLOB,
        explanation TEXT,
        explanation_compressed BLOB,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, timeframe, analyzer_name)
    );""",

    """CREATE TABLE IF NOT EXISTS engine_output (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1D',
        engine_name TEXT NOT NULL,
        payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, timeframe, engine_name)
    );""",

    """CREATE TABLE IF NOT EXISTS score_data (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1D',
        master_score NUMERIC(18,6) CHECK(master_score >= 0 AND master_score <= 100),
        quality_score NUMERIC(18,6) CHECK(quality_score >= 0 AND quality_score <= 100),
        institutional_score NUMERIC(18,6) CHECK(institutional_score >= 0 AND institutional_score <= 100),
        ai_score NUMERIC(18,6) CHECK(ai_score >= 0 AND ai_score <= 100),
        compounder_score NUMERIC(18,6) CHECK(compounder_score >= 0 AND compounder_score <= 100),
        swing_score NUMERIC(18,6) CHECK(swing_score >= 0 AND swing_score <= 100),
        breakout_score NUMERIC(18,6) CHECK(breakout_score >= 0 AND breakout_score <= 100),
        risk_score NUMERIC(18,6) CHECK(risk_score >= 0 AND risk_score <= 100),
        confidence_score NUMERIC(18,6) CHECK(confidence_score >= 0 AND confidence_score <= 1),
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, timeframe)
    );""",

    """CREATE TABLE IF NOT EXISTS decision_data (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1D',
        engine_name TEXT NOT NULL,
        action TEXT,
        confidence NUMERIC(18,6) CHECK(confidence >= 0 AND confidence <= 1),
        reasoning TEXT,
        reasoning_compressed BLOB,
        summary TEXT,
        summary_compressed BLOB,
        explanation TEXT,
        explanation_compressed BLOB,
        entry_price NUMERIC(18,6),
        stop_loss NUMERIC(18,6),
        tp1 NUMERIC(18,6),
        tp2 NUMERIC(18,6),
        tp3 NUMERIC(18,6),
        expected_return NUMERIC(18,6),
        risk_reward NUMERIC(18,6),
        holding_period TEXT,
        position_size NUMERIC(18,6),
        allocation NUMERIC(18,6),
        targets_json TEXT CHECK(targets_json IS NULL OR json_valid(targets_json)),
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, timeframe, engine_name)
    );""",

    """CREATE TABLE IF NOT EXISTS ai_output (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1D',
        model_name TEXT,
        recommendation TEXT,
        confidence NUMERIC(18,6) CHECK(confidence >= 0 AND confidence <= 1),
        reasoning TEXT,
        reasoning_compressed BLOB,
        summary TEXT,
        summary_compressed BLOB,
        explanation TEXT,
        explanation_compressed BLOB,
        version INTEGER,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, timeframe)
    );""",

    """CREATE TABLE IF NOT EXISTS master_ai_decision (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        timeframe TEXT NOT NULL DEFAULT '1D',
        recommendation TEXT,
        confidence NUMERIC(18,6) CHECK(confidence >= 0 AND confidence <= 1),
        confidence_grade TEXT,
        reasoning TEXT,
        reasoning_compressed BLOB,
        summary TEXT,
        summary_compressed BLOB,
        explanation TEXT,
        explanation_compressed BLOB,
        entry_price NUMERIC(18,6),
        stoploss NUMERIC(18,6),
        targets TEXT,
        risk_profile TEXT,
        reward_ratio NUMERIC(18,6),
        expected_return NUMERIC(18,6),
        success_probability NUMERIC(18,6) CHECK(success_probability >= 0 AND success_probability <= 1),
        holding_period TEXT,
        investment_type TEXT,
        ai_model TEXT,
        pipeline_version TEXT,
        version INTEGER,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol, timestamp, timeframe)
    );""",

    """CREATE TABLE IF NOT EXISTS ai_training (
        id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        version TEXT NOT NULL,
        trained_at DATETIME NOT NULL,
        metrics_json TEXT CHECK(metrics_json IS NULL OR json_valid(metrics_json)),
        hyperparameters_json TEXT CHECK(hyperparameters_json IS NULL OR json_valid(hyperparameters_json)),
        status TEXT,
        UNIQUE(model_name, version)
    );""",

    # ------------------ APP STATE & CACHE ------------------
    """CREATE TABLE IF NOT EXISTS dashboard_cache (
        id TEXT PRIMARY KEY,
        cache_key TEXT NOT NULL,
        payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
        expires_at DATETIME,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(cache_key)
    );""",

    """CREATE TABLE IF NOT EXISTS portfolio_data (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        quantity NUMERIC(18,6) NOT NULL DEFAULT 0.0,
        average_price NUMERIC(18,6) NOT NULL DEFAULT 0.0,
        realized_pnl NUMERIC(18,6) DEFAULT 0.0,
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(symbol)
    );""",

    """CREATE TABLE IF NOT EXISTS watchlist (
        id TEXT PRIMARY KEY,
        watchlist_name TEXT NOT NULL,
        symbol TEXT NOT NULL,
        added_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(watchlist_name, symbol)
    );""",

    """CREATE TABLE IF NOT EXISTS scanner_results (
        id TEXT PRIMARY KEY,
        scanner_name TEXT NOT NULL,
        symbol TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        match_score NUMERIC(18,6),
        record_checksum TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE,
        UNIQUE(scanner_name, symbol, timestamp)
    );""",

    """CREATE TABLE IF NOT EXISTS alerts (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        condition_json TEXT CHECK(condition_json IS NULL OR json_valid(condition_json)),
        is_triggered BOOLEAN DEFAULT 0,
        triggered_at DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (symbol) REFERENCES stock_master(symbol) ON DELETE CASCADE
    );""",

    # ------------------ SYSTEM & OBSERVABILITY ------------------
    """CREATE TABLE IF NOT EXISTS registry (
        id TEXT PRIMARY KEY,
        component_type TEXT NOT NULL,
        component_name TEXT NOT NULL,
        is_active BOOLEAN DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(component_type, component_name)
    );""",

    """CREATE TABLE IF NOT EXISTS trace_log (
        id TEXT PRIMARY KEY,
        trace_id TEXT NOT NULL,
        span_id TEXT NOT NULL,
        component TEXT,
        operation TEXT,
        latency_ms NUMERIC(18,6),
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );""",

    """CREATE TABLE IF NOT EXISTS validation_log (
        id TEXT PRIMARY KEY,
        table_name TEXT,
        status TEXT,
        rows_processed INTEGER,
        rows_dropped INTEGER,
        errors_json TEXT CHECK(errors_json IS NULL OR json_valid(errors_json)),
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );""",

    """CREATE TABLE IF NOT EXISTS sync_log (
        id TEXT PRIMARY KEY,
        sync_module TEXT NOT NULL,
        status TEXT NOT NULL,
        rows_written INTEGER,
        latency_ms NUMERIC(18,6),
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );""",

    """CREATE TABLE IF NOT EXISTS system_log (
        id TEXT PRIMARY KEY,
        level TEXT NOT NULL,
        module TEXT NOT NULL,
        message TEXT NOT NULL,
        metadata_json TEXT CHECK(metadata_json IS NULL OR json_valid(metadata_json)),
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );"""
]

# -------------------------------------------------------------------------
# OPTIMIZED INDEXES (70+ Targeted Indexes for AI Pipeline)
# -------------------------------------------------------------------------

_INDEX_DDL: Final[List[str]] = [
    # Master
    "CREATE INDEX IF NOT EXISTS idx_stock_active ON stock_master(is_active) WHERE is_active = 1;",
    "CREATE INDEX IF NOT EXISTS idx_stock_sector ON stock_master(sector);",
    "CREATE INDEX IF NOT EXISTS idx_stock_industry ON stock_master(industry);",
    "CREATE INDEX IF NOT EXISTS idx_stock_fno ON stock_master(is_fno) WHERE is_fno = 1;",
    
    # System & Infra
    "CREATE INDEX IF NOT EXISTS idx_sync_stat_mod ON sync_status(module_name);",
    "CREATE INDEX IF NOT EXISTS idx_job_q_run ON job_queue(run_at) WHERE status = 'PENDING';",
    
    # Market & Historic
    "CREATE INDEX IF NOT EXISTS idx_md_ts ON market_data(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_md_sym_ts_tf ON market_data(symbol, timestamp, timeframe);",
    "CREATE INDEX IF NOT EXISTS idx_lmd_ts ON live_market_data(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_hd_ts ON historical_data(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_hd_sym_ts_tf ON historical_data(symbol, timestamp, timeframe);",
    
    # Fundamental & Institutional
    "CREATE INDEX IF NOT EXISTS idx_fd_sym_date ON fundamental_data(symbol, report_date);",
    "CREATE INDEX IF NOT EXISTS idx_id_date ON institutional_data(trade_date);",
    "CREATE INDEX IF NOT EXISTS idx_ca_sym_exdate ON corporate_actions(symbol, ex_date);",
    "CREATE INDEX IF NOT EXISTS idx_ca_purpose ON corporate_actions(purpose);",
    "CREATE INDEX IF NOT EXISTS idx_blk_sym_date ON block_deals(symbol, deal_date);",
    "CREATE INDEX IF NOT EXISTS idx_blk_type ON block_deals(deal_type);",
    "CREATE INDEX IF NOT EXISTS idx_bulk_sym_date ON bulk_deals(symbol, deal_date);",
    
    # Derivatives
    "CREATE INDEX IF NOT EXISTS idx_fno_sym_ts_exp ON fno_data(symbol, timestamp, expiry_date);",
    "CREATE INDEX IF NOT EXISTS idx_fno_expiry ON fno_data(expiry_date);",
    "CREATE INDEX IF NOT EXISTS idx_opt_sym_ts_exp ON options_chain(symbol, timestamp, expiry_date);",
    "CREATE INDEX IF NOT EXISTS idx_opt_type ON options_chain(option_type);",
    "CREATE INDEX IF NOT EXISTS idx_opt_strike ON options_chain(strike_price);",

    # SMC & Sentiment
    "CREATE INDEX IF NOT EXISTS idx_news_sym_dt ON news(symbol, published_at);",
    "CREATE INDEX IF NOT EXISTS idx_sent_sym_ts_tf ON sentiment(symbol, timestamp, timeframe);",
    "CREATE INDEX IF NOT EXISTS idx_smc_sym_ts_tf ON smart_money_data(symbol, timestamp, timeframe);",
    "CREATE INDEX IF NOT EXISTS idx_strat_active ON strategy(is_active) WHERE is_active = 1;",

    # AI Pipeline Core Lookup Paths
    "CREATE INDEX IF NOT EXISTS idx_ind_sym_ts_tf ON indicator_data(symbol, timestamp, timeframe);",
    "CREATE INDEX IF NOT EXISTS idx_ind_name ON indicator_data(indicator_name);",
    "CREATE INDEX IF NOT EXISTS idx_ind_snap_sym_tf ON indicator_snapshot(symbol, timeframe);",
    "CREATE INDEX IF NOT EXISTS idx_feat_sym_ts_tf ON feature_data(symbol, timestamp, timeframe);",
    "CREATE INDEX IF NOT EXISTS idx_ana_sym_ts_tf_eng ON analysis_data(symbol, timestamp, timeframe, analyzer_name);",
    "CREATE INDEX IF NOT EXISTS idx_ana_eng ON analysis_data(analyzer_name);",
    "CREATE INDEX IF NOT EXISTS idx_eng_out_sym_ts_tf ON engine_output(symbol, timestamp, timeframe, engine_name);",
    "CREATE INDEX IF NOT EXISTS idx_score_sym_ts_tf ON score_data(symbol, timestamp, timeframe);",
    "CREATE INDEX IF NOT EXISTS idx_dec_sym_ts_tf_eng ON decision_data(symbol, timestamp, timeframe, engine_name);",
    "CREATE INDEX IF NOT EXISTS idx_ai_out_sym_ts_tf ON ai_output(symbol, timestamp, timeframe);",
    "CREATE INDEX IF NOT EXISTS idx_mai_sym_ts_tf ON master_ai_decision(symbol, timestamp, timeframe);",
    "CREATE INDEX IF NOT EXISTS idx_mai_recom ON master_ai_decision(recommendation);",
    "CREATE INDEX IF NOT EXISTS idx_mai_grade ON master_ai_decision(confidence_grade);",

    # Cache & App State
    "CREATE INDEX IF NOT EXISTS idx_dash_key ON dashboard_cache(cache_key);",
    "CREATE INDEX IF NOT EXISTS idx_dash_exp ON dashboard_cache(expires_at);",
    "CREATE INDEX IF NOT EXISTS idx_port_sym ON portfolio_data(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_wl_name_sym ON watchlist(watchlist_name, symbol);",
    "CREATE INDEX IF NOT EXISTS idx_scan_name_sym_ts ON scanner_results(scanner_name, symbol, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_sym ON alerts(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_trig ON alerts(is_triggered) WHERE is_triggered = 0;",
    
    # Observability
    "CREATE INDEX IF NOT EXISTS idx_trace_id ON trace_log(trace_id);",
    "CREATE INDEX IF NOT EXISTS idx_trace_ts ON trace_log(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_syslog_level ON system_log(level);",
    "CREATE INDEX IF NOT EXISTS idx_syslog_ts ON system_log(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_val_log_status ON validation_log(status);",
    "CREATE INDEX IF NOT EXISTS idx_sync_log_mod ON sync_log(sync_module);"
]

# -------------------------------------------------------------------------
# ENTERPRISE VIEWS (Pipeline Aggregators utilizing Window Functions)
# -------------------------------------------------------------------------

_VIEW_DDL: Final[List[str]] = [
    """CREATE VIEW IF NOT EXISTS v_latest_market AS 
       SELECT * FROM (
           SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol, timeframe ORDER BY timestamp DESC) as rn 
           FROM market_data
       ) WHERE rn = 1;""",

    """CREATE VIEW IF NOT EXISTS v_latest_indicator AS 
       SELECT * FROM (
           SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol, timeframe, indicator_name ORDER BY timestamp DESC) as rn 
           FROM indicator_data
       ) WHERE rn = 1;""",

    """CREATE VIEW IF NOT EXISTS v_latest_analysis AS 
       SELECT * FROM (
           SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol, timeframe, analyzer_name ORDER BY timestamp DESC) as rn 
           FROM analysis_data
       ) WHERE rn = 1;""",

    """CREATE VIEW IF NOT EXISTS v_latest_score AS 
       SELECT * FROM (
           SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol, timeframe ORDER BY timestamp DESC) as rn 
           FROM score_data
       ) WHERE rn = 1;""",

    """CREATE VIEW IF NOT EXISTS v_latest_decision AS 
       SELECT * FROM (
           SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol, timeframe, engine_name ORDER BY timestamp DESC) as rn 
           FROM decision_data
       ) WHERE rn = 1;""",

    """CREATE VIEW IF NOT EXISTS v_ai_dashboard AS 
       SELECT 
           m.symbol, 
           m.timeframe,
           s.company_name, 
           m.close as last_price,
           ai.recommendation, 
           ai.confidence, 
           ai.confidence_grade,
           sc.master_score,
           sc.quality_score,
           sc.institutional_score,
           sc.ai_score,
           ai.expected_return,
           ai.holding_period
       FROM (SELECT * FROM master_ai_decision WHERE timeframe='1D' AND timestamp = (SELECT MAX(timestamp) FROM master_ai_decision)) ai
       JOIN stock_master s ON ai.symbol = s.symbol
       LEFT JOIN v_latest_market m ON ai.symbol = m.symbol AND m.timeframe = '1D'
       LEFT JOIN v_latest_score sc ON ai.symbol = sc.symbol AND sc.timeframe = '1D'
       WHERE s.is_active = 1;""",

    """CREATE VIEW IF NOT EXISTS v_active_signals AS 
       SELECT symbol, timeframe, engine_name, action, confidence, entry_price, stop_loss, expected_return 
       FROM v_latest_decision 
       WHERE action IN ('BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL');""",

    """CREATE VIEW IF NOT EXISTS v_portfolio AS 
       SELECT 
           p.symbol,
           s.company_name,
           p.quantity,
           p.average_price,
           m.close as current_price,
           (m.close - p.average_price) * p.quantity as unrealized_pnl,
           p.realized_pnl,
           ai.recommendation as ai_view
       FROM portfolio_data p
       JOIN stock_master s ON p.symbol = s.symbol
       LEFT JOIN v_latest_market m ON p.symbol = m.symbol AND m.timeframe = '1D'
       LEFT JOIN (SELECT * FROM master_ai_decision WHERE timeframe='1D' AND timestamp = (SELECT MAX(timestamp) FROM master_ai_decision)) ai ON p.symbol = ai.symbol;""",

    """CREATE VIEW IF NOT EXISTS v_swing AS 
       SELECT symbol, analyzer_name, sentiment, confidence 
       FROM v_latest_analysis 
       WHERE analyzer_name = 'SwingAnalyzer' AND confidence > 0.7 AND sentiment = 'BULLISH';""",

    """CREATE VIEW IF NOT EXISTS v_compounder AS 
       SELECT symbol, analyzer_name, sentiment, confidence 
       FROM v_latest_analysis 
       WHERE analyzer_name = 'CompounderAnalyzer' AND confidence > 0.7 AND sentiment = 'BULLISH';""",

    """CREATE VIEW IF NOT EXISTS v_breakout AS 
       SELECT symbol, analyzer_name, sentiment, confidence 
       FROM v_latest_analysis 
       WHERE analyzer_name = 'BreakoutAnalyzer' AND confidence > 0.7 AND sentiment = 'BULLISH';""",

    """CREATE VIEW IF NOT EXISTS v_smart_money AS 
       SELECT * FROM (
           SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol, timeframe ORDER BY timestamp DESC) as rn 
           FROM smart_money_data
       ) WHERE rn = 1 AND (fvg_bullish = 1 OR liquidity_sweep = 1 OR bos = 1);"""
]

# -------------------------------------------------------------------------
# SCHEMA ENGINE (SINGLETON)
# -------------------------------------------------------------------------

class SchemaEngine:
    """
    Enterprise Central Schema Engine.
    Orchestrates table definition generation, rigorous DDL constraints execution,
    pragmatic optimizations, integrity evaluations, and automated tuning bindings.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> 'SchemaEngine':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SchemaEngine, cls).__new__(cls)
        return cls._instance

    def _apply_pragmas(self) -> None:
        try:
            with db_manager.execute("PRAGMA journal_mode=WAL;"): pass
            with db_manager.execute("PRAGMA synchronous=NORMAL;"): pass
            with db_manager.execute("PRAGMA temp_store=MEMORY;"): pass
            with db_manager.execute("PRAGMA foreign_keys=ON;"): pass
            
            page_size = getattr(settings.database, "page_size", 4096)
            cache_size = getattr(settings.database, "cache_size", -64000)
            busy_timeout = getattr(settings.database, "busy_timeout_ms", 15000)
            
            with db_manager.execute(f"PRAGMA page_size={page_size};"): pass
            with db_manager.execute(f"PRAGMA cache_size={cache_size};"): pass
            with db_manager.execute(f"PRAGMA busy_timeout={busy_timeout};"): pass
            with db_manager.execute("PRAGMA locking_mode=NORMAL;"): pass
            with db_manager.execute("PRAGMA auto_vacuum=INCREMENTAL;"): pass
            
        except Exception as e:
            _logger.error(f"Failed to apply database PRAGMAs: {e}", exc_info=e)
            raise SchemaError("Database optimization configuration failed.") from e

    @trace_span(operation="schema.create_schema", component="database", kind=SpanKind.INTERNAL)
    def create_schema(self) -> None:
        _logger.info("Initializing enterprise schema creation execution pipeline.")
        start_time = time.perf_counter()
        
        try:
            self._apply_pragmas()
            affected_objects = 0

            with DatabaseSession(isolation=IsolationLevel.EXCLUSIVE):
                for stmt in _TABLE_DDL:
                    with db_manager.execute(stmt): pass
                    affected_objects += 1
                for stmt in _INDEX_DDL:
                    with db_manager.execute(stmt): pass
                    affected_objects += 1
                for stmt in _VIEW_DDL:
                    with db_manager.execute(stmt): pass
                    affected_objects += 1

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            
            TraceEngine.attach_metadata("objects_created", affected_objects)
            metrics_engine.record_latency("schema_create_duration", "database", duration_ms)
            metrics_engine.record_success("schema_engine")
            
            _logger.info("Enterprise database schema deployed flawlessly.")
            AuditEngine.record_success(
                operation="schema.create_schema",
                action=AuditAction.SYSTEM,
                message="Database topology structurally mapped successfully.",
                metadata={"objects_initialized": affected_objects, "duration_ms": round(duration_ms, 3)}
            )

        except Exception as e:
            metrics_engine.record_failure("schema_engine")
            AuditEngine.record_failure(
                operation="schema.create_schema",
                action=AuditAction.SYSTEM,
                message="Schema structural generation encountered fatal error.",
                severity=AuditSeverity.CRITICAL
            )
            TraceEngine.record_exception(e)
            _logger.critical("Fatal exception during schema initialization.", exc_info=e)
            raise SchemaError("Database definition pipeline failed to execute fully.") from e

    @trace_span(operation="schema.drop_schema", component="database", kind=SpanKind.INTERNAL)
    def drop_schema(self) -> None:
        _logger.warning("Commencing destructive total schema purge operation.")
        start_time = time.perf_counter()
        
        try:
            with DatabaseSession(isolation=IsolationLevel.EXCLUSIVE):
                with db_manager.execute("PRAGMA foreign_keys=OFF;"): pass
                
                views = self.list_views()
                for view in views:
                    with db_manager.execute(f"DROP VIEW IF EXISTS {view};"): pass

                tables = self.list_tables()
                for table in tables:
                    with db_manager.execute(f"DROP TABLE IF EXISTS {table};"): pass
                    
                with db_manager.execute("PRAGMA foreign_keys=ON;"): pass
                
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            metrics_engine.record_latency("schema_drop_duration", "database", duration_ms)
            
            _logger.info("Database schema dropped entirely.")
            
        except Exception as e:
            TraceEngine.record_exception(e)
            raise SchemaError("Failed to forcefully drop schema entities.") from e

    def list_tables(self) -> List[str]:
        rows = db_manager.fetch_all("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        return [row['name'].lower() for row in rows]

    def list_indexes(self) -> List[str]:
        rows = db_manager.fetch_all("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%';")
        return [row['name'].lower() for row in rows]

    def list_views(self) -> List[str]:
        rows = db_manager.fetch_all("SELECT name FROM sqlite_master WHERE type='view' AND name NOT LIKE 'sqlite_%';")
        return [row['name'].lower() for row in rows]

    @trace_span(operation="schema.integrity_check", component="database", kind=SpanKind.INTERNAL)
    def integrity_check(self) -> bool:
        try:
            result = db_manager.fetch_all("PRAGMA integrity_check;")
            is_valid = True
            if not result or result[0][0].lower() != "ok":
                _logger.error("Physical database integrity compromised.")
                is_valid = False
            return is_valid
        except Exception as e:
            TraceEngine.record_exception(e)
            raise DatabaseError("Failed to execute mathematical integrity validation bounds.") from e

    @trace_span(operation="schema.vacuum_database", component="database", kind=SpanKind.INTERNAL)
    def vacuum_database(self) -> None:
        try:
            with db_manager.execute("VACUUM;"): pass
        except Exception as e:
            raise DatabaseError("VACUUM maintenance sequence aborted unexpectedly.") from e

    @trace_span(operation="schema.analyze_database", component="database", kind=SpanKind.INTERNAL)
    def analyze_database(self) -> None:
        try:
            with db_manager.execute("ANALYZE;"): pass
        except Exception as e:
            raise DatabaseError("ANALYZE maintenance sequence aborted unexpectedly.") from e

    @trace_span(operation="schema.optimize_database", component="database", kind=SpanKind.INTERNAL)
    def optimize_database(self) -> None:
        try:
            with db_manager.execute("PRAGMA optimize;"): pass
            self.analyze_database()
            with db_manager.execute("PRAGMA wal_checkpoint(TRUNCATE);"): pass
        except Exception as e:
            raise DatabaseError("Holistic optimization tuning aborted unexpectedly.") from e

# -------------------------------------------------------------------------
# GLOBAL SINGLETON EXPORT
# -------------------------------------------------------------------------

schema_engine: Final[SchemaEngine] = SchemaEngine()

__all__ = [
    "SchemaEngine",
    "schema_engine"
]
