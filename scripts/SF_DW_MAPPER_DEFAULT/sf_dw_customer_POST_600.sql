INSERT OVERWRITE INTO delta_stage_customer (${columns:customer::%1$s})

SELECT
   ${columns:customer::c.%1$s AS %1$s}
FROM (
   SELECT ${columns:customer:~c_emailHash:c.%1$s AS %1$s}
      ,CASE WHEN COALESCE(TRIM(Email), '') <> '' THEN SHA2_HEX(LOWER(TRIM(Email)))
           ELSE c_emailHash
           END AS c_emailHash
   FROM delta_udm_customer c
) c
;