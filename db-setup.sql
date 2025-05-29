
create or replace procedure sandbox.COMPARE_MISMATCH_IGNORE_EXCLUDED_COL1(HIVE_SCHEMA String, SF_SCHEMA String, T1 String, T2 String, ID_COL STRING, ID_VAL STRING)
returns ARRAY
language JAVASCRIPT
AS $$
   
    var cols = []
    const colRst = snowflake.execute({sqlText: `desc table ${T1}`})
    while(colRst.next()){
        cols.push(colRst.getColumnValueAsString(1));
    }
       
    let rst = snowflake.execute({
        sqlText: `select * from ${HIVE_SCHEMA}.${T1} where ${ID_COL}='${ID_VAL}'`
    })
    let resT1 = [];
    while(rst.next()){
          
        resT1 = cols.map(c => [c, rst.getColumnValueAsString(c)])
    }
       
    rst = snowflake.execute({
        sqlText: `select * from ${SF_SCHEMA}.${T2} where ${ID_COL}='${ID_VAL}'`
    })
    let res = []
     while(rst.next()){
        res = resT1.filter(r => rst.getColumnValueAsString(r[0]) != r[1]).map(r => [...r, rst.getColumnValueAsString(r[0])])
    }
    let excludeColumns = getExcludeColumns();
    let excludeColumnsArr = excludeColumns.split(",");
    const excludeColumnsMap = new Map(excludeColumnsArr.map(s => [s,1]));
 
    let r = res.filter(arr => !excludeColumnsMap.has(arr[0]));
    return r;
       
    function getExcludeColumns(){
         var statement = snowflake.createStatement(
             {
                 sqlText: `SELECT EXCLUDE_COLUMNS FROM DW.DW_SHADOW_RUN_CONF WHERE TABLE_NAME='${T1}';`
             }
         );
         var result_set = statement.execute();
         var bi_summary_table_list =[];
         var excludeColumns = "";
         while(result_set.next()){
             excludeColumns = result_set.getColumnValue(1);
         }
         return excludeColumns.toString();
    }
$$;


create or replace table customer_rk clone dw.delta_stage_customer;

use role a1sf_role_101_676_acquia_user;
show grants on table sandbox.delta_stage_customer_rk;
create or replace table sandbox.delta_stage_customer_rk clone dw.delta_stage_customer;

select * from sandbox.delta_stage_customer_rk where id='FTP_CSV_11_67E7A4881A42DC7A419D1332E7';

select * from sandbox.delta_stage_customer_rk limit 1;

select * from sandbox.dw_data_metrics;
update sandbox.delta_stage_customer_rk set LASTname='ramirezee' where id='FTP_CSV_11_67E7A4881A42DC7A419D1332E7';


select current_database(), current_schema(), current_role();

update SANDBOX.dw_data_metrics set data_discrepancy_pk_values = ARRAY_CONSTRUCT('FTP_CSV_11_67E7A4881A42DC7A419D1332E7') where table_name='CUSTOMER';

call COMPARE_MISMATCH_IGNORE_EXCLUDED_COL1('SANDBOX', 'SANDBOX', 'DELTA_STAGE_CUSTOMER_RK', 'CUSTOMER_RK', 'ID', 'FTP_CSV_11_67E7A4881A42DC7A419D1332E7');

SELECT EXCLUDE_COLUMNS FROM DW.DW_SHADOW_RUN_CONF WHERE TABLE_NAME='CUSTOMER';


INSERT INTO SANDBOX.dw_data_metrics (
    ID,
    TABLE_NAME,
    PRIMARY_COLUMN,
    TOTAL_RECORD_COUNT_HIVE,
    TOTAL_RECORD_COUNT_SF,
    ROW_COUNT_ONLY_IN_HIVE,
    HIVE_ONLY_PK_VALUES,
    ROW_COUNT_ONLY_IN_SF,
    SF_ONLY_PK_VALUES,
    ROW_COUNT_DATA_DISCREPANCY,
    DATA_DISCREPANCY_PK_VALUES,
    ROW_CREATED
)
SELECT
    2,
    'TRANSACTION',
    'ID',
    408623,
    408653,
    1,
    ARRAY_CONSTRUCT('FTP_CSV_12_62d9503ff34d203e08b6a442'),
    31,
    ARRAY_CONSTRUCT('KFK_0_00311172'),
    375448,
    ARRAY_CONSTRUCT('FTP_CSV_11_67E7A4881A42DC7A419D1332E7'),
    '2024-07-26 22:15:03.798';

