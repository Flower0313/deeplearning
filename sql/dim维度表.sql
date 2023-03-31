drop table dim_stock;
create table if not exists dim_stock
(
    `code`                 STRING COMMENT '股票代码',
    `name`                 STRING COMMENT '股票名称',
    `closing_price`        decimal(10, 2) COMMENT '今日收盘',
    `closing_diff`         decimal(10, 2) COMMENT '今日涨额',
    `deal_amount`          decimal(20, 5) COMMENT '成交额',
    `highest`              decimal(10, 5) COMMENT '最高价',
    `lowest`               decimal(10, 5) COMMENT '最低价',
    `opening_price`        decimal(10, 2) COMMENT '今日开盘',
    `market`               BIGINT COMMENT '市场',
    `up_down_rate`         DECIMAL(10, 2) COMMENT '涨跌幅',
    `up_down_rate5`        DECIMAL(10, 2) COMMENT '5日涨幅',
    `up_down_rate10`       DECIMAL(10, 2) COMMENT '10日涨幅',
    `up_down_amount`       DECIMAL(16, 2) COMMENT '涨跌额',
    `turnover_rate`        DECIMAL(5, 2) COMMENT '换手率',
    `PE_ratio_d`           DECIMAL(32, 2) COMMENT '市盈率(动态)',
    `amplitude`            DECIMAL(10, 2) COMMENT '振幅',
    `volume_ratio`         DECIMAL(10, 2) COMMENT '量比',
    `t_1_price`            DECIMAL(10, 2) COMMENT '昨日收盘',
    `total_market_v`       DECIMAL(32, 5) COMMENT '总市值',
    `circulation_market_v` DECIMAL(32, 5) COMMENT '流通市值',
    `price_to_b_ratio`     DECIMAL(32, 5) COMMENT '市净率',
    `increase_this_year`   DECIMAL(10, 2) COMMENT '今年涨幅',
    `time_to_market`       BIGINT COMMENT '上市时间',
    `outer_disk`           DECIMAL(32, 2) COMMENT '外盘',
    `inner_disk`           DECIMAL(32, 2) COMMENT '内盘',
    `roe`                  DECIMAL(20, 3) COMMENT 'ROE加权净资产收益率',
    `total_share_capital`  DECIMAL(32, 3) COMMENT '总股本',
    `tradable_shares`      DECIMAL(32, 3) COMMENT '流通A股',
    `total_revenue`        DECIMAL(32, 5) COMMENT '总营收',
    `total_revenue_r`      DECIMAL(32, 5) COMMENT '总营收同比',
    `gross_profit_margin`  DECIMAL(32, 10) COMMENT '毛利率',
    `total_assets`         DECIMAL(32, 10) COMMENT '总资产',
    `debt_ratio`           DECIMAL(32, 10) COMMENT '负债率',
    `industry`             STRING COMMENT '行业',
    `regional_plate`       STRING COMMENT '地区板块',
    `profit`               DECIMAL(32, 10) COMMENT '收益',
    `PE_ratio_s`           DECIMAL(32, 2) COMMENT '市盈率(静态)',
    `ttm`                  DECIMAL(32, 2) COMMENT '市盈率(TTM)',
    `net_assets`           DECIMAL(32, 10) COMMENT '净资产',
    `deal_vol`             DECIMAL(20, 5) COMMENT '成交量',
    `dealTradeStae`        BIGINT COMMENT '交易状态',
    `commission`           DECIMAL(5, 2) COMMENT '委比',
    `net_margin`           DECIMAL(32, 5) COMMENT '净利率',
    `total_profit`         decimal(32, 5) COMMENT '总利润',
    `net_assets_per_share` decimal(32, 5) COMMENT '每股净资产',
    `net_profit`           decimal(32, 5) COMMENT '净利润',
    `net_profit_r`         decimal(32, 5) COMMENT '净利润同比',
    `unearnings_per_share` decimal(32, 5) COMMENT '每股未分配利润',
    `main_inflow`          decimal(32, 5) COMMENT '主力净流入',
    `main_inflow_ratio`    decimal(20, 2) COMMENT '主力比',
    `Slarge_inflow`        decimal(32, 5) COMMENT '超大单净流入',
    `Slarge_inflow_ratio`  decimal(20, 2) COMMENT '超大单净比',
    `large_inflow`         decimal(32, 5) COMMENT '大单净流入',
    `large_inflow_ratio`   decimal(20, 2) COMMENT '大单净比',
    `mid_inflow`           decimal(32, 5) COMMENT '中单净流入',
    `mid_inflow_ratio`     decimal(20, 2) COMMENT '中单净比',
    `small_inflow`         decimal(32, 5) COMMENT '小单净流入',
    `small_inflow_ratio`   decimal(20, 2) COMMENT '小单净比',
    `asi`                  decimal(20, 5) COMMENT 'asi',
    `bbi`                  decimal(20, 5) COMMENT 'bbi',
    `br`                   decimal(20, 5) COMMENT 'br',
    `ar`                   decimal(20, 5) COMMENT 'ar',
    `ma3`                  decimal(20, 5) COMMENT 'ma3',
    `ma5`                  decimal(20, 5) COMMENT 'ma5',
    `ma6`                  decimal(20, 5) COMMENT 'ma5',
    `ma10`                 decimal(20, 5) COMMENT 'ma10',
    `ma12`                 decimal(20, 5) COMMENT 'ma12',
    `ma20`                 decimal(20, 5) COMMENT 'ma20',
    `ma24`                 decimal(20, 5) COMMENT 'ma24',
    `ma50`                 decimal(20, 5) COMMENT 'ma50',
    `ma60`                 decimal(20, 5) COMMENT 'ma60',
    `bias6`                decimal(20, 5) COMMENT 'bias6',
    `bias12`               decimal(20, 5) COMMENT 'bias12',
    `bias24`               decimal(20, 5) COMMENT 'bias24',
    `bias36`               decimal(20, 5) COMMENT 'bias36',
    `mtr`                  decimal(20, 5) COMMENT 'mtr',
    `atr`                  decimal(20, 5) COMMENT 'atr',
    `dpo`                  decimal(20, 5) COMMENT 'dpo',
    `upper_ene`            decimal(20, 5) COMMENT 'upper_ene',
    `lower_ene`            decimal(20, 5) COMMENT 'lower_ene',
    `ene`                  decimal(20, 5) COMMENT 'ene',
    `emv`                  decimal(20, 5) COMMENT 'emv',
    `mtm`                  decimal(20, 5) COMMENT 'mtm',
    `wr6`                  decimal(20, 5) COMMENT 'wr6',
    `wr10`                 decimal(20, 5) COMMENT 'wr10',
    `psy`                  decimal(20, 5) COMMENT 'psy',
    `psyma`                decimal(20, 5) COMMENT 'psyma',
    `roc`                  decimal(20, 5) COMMENT 'roc',
    `maroc`                decimal(20, 5) COMMENT 'maroc',
    `upperl`               decimal(20, 5) COMMENT 'upperl',
    `uppers`               decimal(20, 5) COMMENT 'uppers',
    `lowerl`               decimal(20, 5) COMMENT 'lowerl',
    `lowers`               decimal(20, 5) COMMENT 'lowers',
    `k`                    decimal(20, 5) COMMENT 'k',
    `d`                    decimal(20, 5) COMMENT 'd',
    `j`                    decimal(20, 5) COMMENT 'j',
    `pdi`                  decimal(20, 5) COMMENT 'pdi',
    `mdi`                  decimal(20, 5) COMMENT 'mdi',
    `adx`                  decimal(20, 5) COMMENT 'adx',
    `dif`                  decimal(32, 10) COMMENT 'dif',
    `dea`                  decimal(20, 5) COMMENT 'dea',
    `macd`                 decimal(20, 5) COMMENT 'macd',
    `rsi6`                 decimal(20, 5) COMMENT 'rsi6',
    `rsi12`                decimal(20, 5) COMMENT 'rsi12',
    `rsi24`                decimal(20, 5) COMMENT 'rsi24',
    `sar`                  decimal(20, 5) COMMENT 'sar',
    `trix`                 decimal(20, 5) COMMENT 'trix',
    `lwr1`                 decimal(20, 5) COMMENT 'lwr1',
    `lwr2`                 decimal(20, 5) COMMENT 'lwr2',
    `stor`                 decimal(20, 5) COMMENT '服务于mike',
    `midr`                 decimal(20, 5) COMMENT '服务于mike',
    `wekr`                 decimal(20, 5) COMMENT '服务于mike',
    `weks`                 decimal(20, 5) COMMENT '服务于mike',
    `mids`                 decimal(20, 5) COMMENT '服务于mike',
    `stos`                 decimal(20, 5) COMMENT '服务于mike',
    `obv`                  decimal(20, 5) COMMENT 'obv',
    `cci`                  decimal(20, 5) COMMENT 'cci',
    `boll`                 decimal(20, 5) COMMENT 'boll',
    `boll_up`              decimal(20, 5) COMMENT 'boll_up',
    `boll_down`            decimal(20, 5) COMMENT 'boll_down',
    `ds`                   string COMMENT '交易日'
) COMMENT '东方财富A股维度表'
    PARTITIONED BY (`dt` string)
    ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
    STORED AS ORC
    LOCATION '/hive/warehouse/df_db/dim/dim_stock'
    TBLPROPERTIES ('orc.compress' = 'snappy');

-- 每日同步
insert into table dim_stock partition (dt = '2022-09-02')
select ods.`code`,
       ods.`name`,
       `closing_price`,
       `closing_diff`,
       ods.`deal_amount`,
       ods.`highest`,
       ods.`lowest`,
       ods.`opening_price`,
       `market`,
       `up_down_rate`,
       `up_down_rate5`,
       `up_down_rate10`,
       `up_down_amount`,
       `turnover_rate`,
       `PE_ratio_d`,
       `amplitude`,
       `volume_ratio`,
       `t_1_price`,
       `total_market_v`,
       `circulation_market_v`,
       `price_to_b_ratio`,
       `increase_this_year`,
       `time_to_market`,
       `outer_disk`,
       `inner_disk`,
       `roe`,
       `total_share_capital`,
       `tradable_shares`,
       `total_revenue`,
       `total_revenue_r`,
       `gross_profit_margin`,
       `total_assets`,
       `debt_ratio`,
       `industry`,
       `regional_plate`,
       `profit`,
       `PE_ratio_s`,
       `ttm`,
       `net_assets`,
       `deal_vol`,
       `dealTradeStae`,
       `commission`,
       `net_margin`,
       `total_profit`,
       `net_assets_per_share`,
       `net_profit`,
       `net_profit_r`,
       `unearnings_per_share`,
       `main_inflow`,
       `main_inflow_ratio`,
       `Slarge_inflow`,
       `Slarge_inflow_ratio`,
       `large_inflow`,
       `large_inflow_ratio`,
       `mid_inflow`,
       `mid_inflow_ratio`,
       `small_inflow`,
       `small_inflow_ratio`,
       `asi`,
       `bbi`,
       `br`,
       `ar`,
       `ma3`,
       `ma5`,
       `ma6`,
       `ma10`,
       `ma12`,
       `ma20`,
       `ma24`,
       `ma50`,
       `ma60`,
       `bias6`,
       `bias12`,
       `bias24`,
       `bias36`,
       `mtr`,
       `atr`,
       `dpo`,
       `upper_ene`,
       `lower_ene`,
       `ene`,
       `emv`,
       `mtm`,
       `wr6`,
       `wr10`,
       `psy`,
       `psyma`,
       `roc`,
       `maroc`,
       `upperl`,
       `uppers`,
       `lowerl`,
       `lowers`,
       `k`,
       `d`,
       `j`,
       `pdi`,
       `mdi`,
       `adx`,
       `dif`,
       `dea`,
       `macd`,
       `rsi6`,
       `rsi12`,
       `rsi24`,
       `sar`,
       `trix`,
       `lwr1`,
       `lwr2`,
       `stor`,
       `midr`,
       `wekr`,
       `weks`,
       `mids`,
       `stos`,
       `obv`,
       `cci`,
       `boll`,
       `boll_up`,
       `boll_down`,
       ods.`ds`
from (select * from dwd_stock_detail where dt = '2022-09-02') dwd
         inner join (select * from ods_a_stock_detail_day where dt = '2022-09-02') ods
                    on dwd.code = ods.code;