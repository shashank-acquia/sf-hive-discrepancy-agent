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