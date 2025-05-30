import os
import json
import logging
from dotenv import load_dotenv
from typing import Dict
from langchain_community.chat_models import ChatOpenAI
from typing import Dict, Union
from langchain_core.prompts import PromptTemplate
import re

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
SCRIPT_DIR = os.getenv("SCRIPT_DIR")
table_name = os.getenv("TARGET_TABLE", "CUSTOMER")
metadata_dir = os.getenv("METADATA_DIR", "resources/prod-gcp")

if not os.path.isdir(metadata_dir):
    logger.warning(f"Metadata directory not found: {metadata_dir}")
    logger.warning("Script expansion may not work correctly without metadata.")

# Load LLM
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4.1", cache=False)

# Full Conversion Rules
conversion_rules = """
Please convert a sql script written using hive on spark syntax into a Snowflake compatible syntax using below rules:
1. Replace all occurrences of dw table that has the convention udm_{{entity}} with delta_stage_{{entity}} where {{entity}} is the name of the table. Do this ONLY IF the dw table occurrence is in a DDL statement like INSERT or INSERT OVERWRITE.
2a. If you encounter INSERT statements for temporary tables, keep them as-is
2b. If the dw table shows up in a FROM or JOIN statement, replace it with PUBLIC.{{entity}} where {{entity}} is the name of the table.
3. Replace all occurrences of pv table that has the convention udm_pv_{{entity}} with PUBLIC.{{entity}} where {{entity}} is the name of the table. Do this ONLY if the pv table occurrence is in a FROM or a JOIN statement.
4. Replace all occurrences of  sparse table that has the convention udm_s_{{entity}} with delta_udm_{{entity}} where {{entity}} is the name of the table
5. For insert statements, at the end, append the column list - (${{columns:{{entity}}::%1$s}}) where {{entity}} is the name of the table. The insert statement will always be to a dw table so refer the entity name from rule 1 above.
   Don't add any value to shorthand , just keep  in this format : (${{columns:{{entity}}::%1$s}})
6. Please retain all the commented lines as-is. A comment starts with a '--' symbol.
7. Replace the function call to 'locate' with 'position'
8. Replace the reflect function call using URLDecoder to use the custom function 'decodeurl' instead. For example, reflect("java.net.URLDecoder", "decode", "VALUE") or 
 reflect("java.net.URLDecoder", "decode", "VALUE","UTF-8")  should become decodeurl("VALUE").
9. Replace the sha256hex conversion to use the custom function 'SHA2_HEX' instead. For example, reflect("org.apache.commons.codec.digest.DigestUtils", "sha256Hex", "VALUE") should become SHA2_HEX("VALUE")
10. Replace hash function call with hashcode. For example, hash("VALUE") should become hashcode("VALUE")
11. Replace LCASE function call with LOWER. For example, LCASE("VALUE") should become LOWER("VALUE")
12. Replace INSTR function call with position. For example, INSTR("VALUE", {{character}}) should become position({{character}},"VALUE")
13. Replace 'CASE WHEN map_values (collect_max_one (COALESCE(%1$s,''),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0]= 0L THEN NULL
                    ELSE map_keys (collect_max_one (COALESCE(%1$s,''),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] END AS %1$s:
                CASE WHEN map_values (collect_max_one (COALESCE(%1$s,false),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0]= 0L THEN NULL
                    ELSE map_keys (collect_max_one (COALESCE(%1$s,false),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] END AS %1$s:
                CASE WHEN map_values (collect_max_one (COALESCE(%1$s,0),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0]= 0L THEN NULL
                    ELSE map_keys (collect_max_one (COALESCE(%1$s,0),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] END AS %1$s:
                CASE WHEN map_values (collect_max_one (COALESCE(%1$s,cast(0 as double)),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] = 0L THEN NULL
                    ELSE map_keys (collect_max_one (COALESCE(%1$s,cast(0 as double)),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] END AS %1$s:
                CASE WHEN map_values (collect_max_one ( COALESCE(cast(%1$s as string),'') ,CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] = 0L THEN NULL
                    ELSE cast(map_keys (collect_max_one (COALESCE(cast(%1$s as string),'') ,CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] as decimal) END AS %1$s' code block with 
    'SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),IFF(COALESCE(%1$s,'') = '',NULL,%1$s))),14) AS %1$s:
             CAST(SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),%1$s)),14)  AS boolean) AS %1$s:
             SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),%1$s)),14) AS %1$s:
             CAST(SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),%1$s)),14) AS double) AS %1$s:
             CAST(SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),%1$s)),14) AS double) AS %1$s'
14.Replace UNIX_TIMESTAMP()*1000 calcultion with CURRENT_TIMESTAMP()
15. On main scripts only, if "SET MD5Source=" statement is missing on the script, please add "SET MD5Source='PUBLIC.{{}}'"  top of the script and replcace {{}} with dw table name IF it is main script. It is a main script if it PRECISELY contains: SUBSTRING(MAX(CONCAT(RowModified, IFF(COALESCE(%1$s, '') = '', NULL, %1$s))), 14) AS %1$s: or collect_max_one
16  Replace "LEFT OUTER JOIN PUBLIC.{{}}" with 'LEFT OUTER JOIN TABLE($MD5Source)'.
16  Replace "LEFT OUTER JOIN udm_{{entity}}" with LEFT OUTER JOIN TABLE($MD5Source)
16  Replace PUBLIC.{{entity}} with TABLE($MD5Source) if entity is dw table name
17. Do not replace "md5" function with no other hashing function.
18. Do not add any comment lines if it doesn't exist in the given script.
19. Remove MAPJOIN hint.
20. Please remove 'SET mapred.reduce.tasks' assigment.
21. Replace the function SPLIT({{ColumName}},'_')[{{Number}}] to SPLIT_PART({{ColumName}},'_',{{Number}}).
22. Replace "${{hiveconf:Variable}}"" to "$Variable"
23. Replace the variable assignment  from "SET  Variable = ('a', 'b'); "  to "SET  Variable = 'a,b'; "
25. Replace double quotes with single quotes
26. Replace 'IF' function to 'IFF'
27. Replace '                SUBSTRING(MAX(CONCAT(RowModified,IF(COALESCE(%1$s,'') = '',NULL,%1$s))),14) AS %1$s:
                CAST(CAST(SUBSTRING(MAX(CONCAT(RowModified,CAST(%1$s AS int))),14) AS int) AS boolean) AS %1$s:
                SUBSTRING(MAX(CONCAT(RowModified,%1$s)),14) AS %1$s:
                CAST(SUBSTRING(MAX(CONCAT(RowModified,%1$s)),14) AS double) AS %1$s:
                CAST(SUBSTRING(MAX(CONCAT(RowModified,%1$s)),14) AS decimal(18,4)) AS %1$s' TO 'SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),IFF(COALESCE(%1$s,'') = '',NULL,%1$s))),14) AS %1$s:
                CAST(SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),%1$s)),14) AS boolean) AS %1$s:
                SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),%1$s)),14) AS %1$s:
                CAST(SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),%1$s)),14) AS double) AS %1$s:
                CAST(SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),%1$s)),14) AS double) AS %1$s'
28. Replace INSERT dw table statment to INSERT OVERWRITE INTO
29. Replace 'tbl' table create statement to CREATE OR REPLACE TEMPORARY TABLE
30. Don't replace table aliases even it is udm ,sparse or dw 
31. Do not make any change on the shorthand scripts which is defined 5. If there is any {{aliases}} in the shorthand (${{columns:::{{aliases}}%1$s}}) , kept it as is.
32. Remove "* 1000" for conversion for Date columns 
33. Do not make any change on the query structure,keep aliases, sub-select as it is.
34. Replace RowModified + 1 with DATEADD(second, 1, sc.RowModified)
35. FIND_IN_SET should remain and should not be replaced with the native ARRAY_CONTAINS
36. to_date(from_unixtime(floor(<some_date_column>/1000))) should be modified to DATE(<some_date_column>)
37.GIVE THE EXACT CONVERTED SCRIPT WITH SMALL EXPLANATION AS COMMENT , SCRIPT SHOULD BE PRODUCTION READY TO COPY AND PASTE
Here are two examples.   

Example 1 (not a main script) Before:
```
INSERT INTO TABLE udm_customer


SELECT
	${columns:customer::c.%1$s AS %1$s}
FROM (

	SELECT ${columns:customer:~c_emailHash:c.%1$s AS %1$s}
		,CASE WHEN COALESCE(TRIM(Email),'') <> ''  THEN reflect('org.apache.commons.codec.digest.DigestUtils', 'sha256Hex', lower(TRIM(Email))) 
           ELSE c_emailHash 
           END AS c_emailHash
	FROM udm_s_customer c

) c
;
```

Example 1 (not a main script) After:
```
INSERT OVERWRITE INTO delta_stage_customer (${columns::customer::%1$s})

SELECT
	${columns:customer::c.%1$s AS %1$s}
FROM (
	SELECT ${columns:customer:~c_emailHash:c.%1$s AS %1$s}
		,CASE WHEN COALESCE(TRIM(Email),'') <> '' THEN SHA2_HEX(LOWER(TRIM(Email)))
           ELSE c_emailHash
           END AS c_emailHash
	FROM delta_udm_customer c
) c
;
```

Example 2 (main script) Before:
```
INSERT INTO TABLE udm_c_currency
SELECT ${columns:c_currency::t.%1$s AS %1$s}
FROM (
    SELECT ${columns:c_currency:~RowCreated:COALESCE(sparse.%1$s,dw.%1$s) AS %1$s}
        ,COALESCE(dw.RowCreated,sparse.RowCreated) AS RowCreated
    FROM (
        SELECT
             Sourcec_currencyNumber AS ID
            , Sourcec_currencyNumber
            ,${columns:c_currency:~ID,Sourcec_currencyNumber,RowCreated,RowModified,Batch:
                CASE WHEN map_values (collect_max_one (COALESCE(%1$s,''),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0]= 0L THEN NULL
                    ELSE map_keys (collect_max_one (COALESCE(%1$s,''),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] END AS %1$s:
                CASE WHEN map_values (collect_max_one (COALESCE(%1$s,false),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0]= 0L THEN NULL
                    ELSE map_keys (collect_max_one (COALESCE(%1$s,false),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] END AS %1$s:
                CASE WHEN map_values (collect_max_one (COALESCE(%1$s,0),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0]= 0L THEN NULL
                    ELSE map_keys (collect_max_one (COALESCE(%1$s,0),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] END AS %1$s:
                CASE WHEN map_values (collect_max_one (COALESCE(%1$s,cast(0 as double)),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] = 0L THEN NULL
                    ELSE map_keys (collect_max_one (COALESCE(%1$s,cast(0 as double)),CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] END AS %1$s:
                CASE WHEN map_values (collect_max_one ( COALESCE(cast(%1$s as string),'') ,CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] = 0L THEN NULL
                    ELSE cast(map_keys (collect_max_one (COALESCE(cast(%1$s as string),'') ,CASE WHEN %1$s IS NULL then 0L else RowModified END))[0] as decimal(18,4)) END AS %1$s
            }
            ,MIN(RowCreated) AS RowCreated
            ,MAX(RowModified) AS RowModified
            ,MAX(Batch) As Batch
        FROM (
            SELECT ${columns:c_currency::c1.%1$s AS %1$s}
                , ROW_NUMBER() OVER (PARTITION BY c_conversionDate,c_toCurrency,c_originCurrency ORDER BY c_dateModified DESC) AS Rank
            FROM udm_s_c_currency c1
            WHERE COALESCE(c_toCurrency,'')= 'USD'
        ) t
        WHERE t.Rank = 1
        GROUP BY  Sourcec_currencyNumber
    ) sparse
    LEFT OUTER JOIN udm_c_currency dw on sparse.ID = dw.ID
    WHERE md5(concat(${columns:c_currency:~TenantId,SourceSystemID,Source,Batch,RowCreated,RowModified,ID:COALESCE(CAST(sparse.%1$s AS STRING),'NULL')})) !=
         md5(concat(${columns:c_currency:~TenantId,SourceSystemID,Source,Batch,RowCreated,RowModified,ID:COALESCE(CAST(dw.%1$s AS STRING),'NULL')}))
) t
;
```

Example 2 (main script) After:
```
-- Added MD5Source statement as the script contains the main script pattern
SET MD5Source='PUBLIC.c_currency'; -- Table name derived from udm_c_currency

INSERT OVERWRITE INTO delta_stage_c_currency (${columns:c_currency::%1$s})
SELECT ${columns:c_currency::t.%1$s AS %1$s}
FROM (
    SELECT ${columns:c_currency:~RowCreated:COALESCE(sparse.%1$s,dw.%1$s) AS %1$s}
        ,COALESCE(dw.RowCreated,sparse.RowCreated) AS RowCreated
    FROM (
        SELECT
            Sourcec_currencyNumber AS ID
            , Sourcec_currencyNumber
            ,${columns:c_currency:~ID,Sourcec_currencyNumber,RowCreated,RowModified,Batch:
                SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),IFF(COALESCE(%1$s,'') = '',NULL,%1$s))),14) AS %1$s:
                CAST(SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),%1$s)),14)  AS boolean) AS %1$s:
                SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),%1$s)),14) AS %1$s:
                CAST(SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),%1$s)),14) AS double) AS %1$s:
                CAST(SUBSTRING(MAX(CONCAT(TO_NUMBER_CUSTOM(RowModified),%1$s)),14) AS double) AS %1$s
            }
            ,MIN(RowCreated) AS RowCreated
            ,MAX(RowModified) AS RowModified
            ,MAX(Batch) As Batch
        FROM (
            SELECT ${columns:c_currency::c1.%1$s AS %1$s}
                , ROW_NUMBER() OVER (PARTITION BY c_conversionDate,c_toCurrency,c_originCurrency ORDER BY c_dateModified DESC) AS Rank
            FROM delta_udm_c_currency c1
            WHERE COALESCE(c_toCurrency,'')= 'USD'
        ) t
        WHERE t.Rank = 1
        GROUP BY Sourcec_currencyNumber
    ) sparse
    LEFT OUTER JOIN TABLE($MD5Source) dw on sparse.ID = dw.ID
    WHERE md5(concat(${columns:c_currency:~TenantId,SourceSystemID,Source,Batch,RowCreated,RowModified,ID:COALESCE(CAST(sparse.%1$s AS STRING),'NULL')})) !=
         md5(concat(${columns:c_currency:~TenantId,SourceSystemID,Source,Batch,RowCreated,RowModified,ID:COALESCE(CAST(dw.%1$s AS STRING),'NULL')}))
) t
;
```

"""


def escape_curly_braces(text: str) -> str:
    # Escape all {placeholder} that are NOT already escaped (i.e., not already {{...}})
    return re.sub(r'(?<!{){([^{}]+)}(?!})', r'{{\1}}', text)

def ask_llm_to_fix(hive_sql_script: str) -> str:
    logger.info("🚀 Invoking LLM to convert Hive script...")
    logger.info(escape_curly_braces(conversion_rules));

    escaped_rules = escape_curly_braces(conversion_rules)

    prompt_template = PromptTemplate.from_template(
    "hive script is:\n{input_data}\n\n" + escape_curly_braces(conversion_rules)
     )
    prompt = prompt_template.format(input_data=hive_sql_script)

    response = llm.invoke(prompt)
    logger.info("✅ LLM response received")
    return response.content

def script_converter(input_data: Union[str, Dict[str, str]]) -> str:
    logger.info("⚙️ Running script_converter...")

    if isinstance(input_data, str):
        try:
            data = json.loads(input_data)
        except json.JSONDecodeError as e:
            logger.error("Input is not valid JSON.")
            raise ValueError("Expected JSON string with a 'script' key") from e
    elif isinstance(input_data, dict):
        data = input_data
    else:
        raise TypeError("Input must be a JSON string or a dictionary")

    sql_script = data.get("script")
    if not sql_script:
        raise KeyError("Missing 'script' key in input data")

    logger.info(f"🧪 Converting script with length: {len(sql_script)} characters")

    converted_script = ask_llm_to_fix(sql_script)

    return json.dumps({
        "results": converted_script
    }, indent=2)

