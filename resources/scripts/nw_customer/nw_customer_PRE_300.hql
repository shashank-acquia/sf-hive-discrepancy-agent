-- A1CS-4363, to populate c_channelViewKeyCode for ChannelView integration
-- Get the DW customer records and populate the ChannelViewKeyCode field if there is a match 

INSERT INTO TABLE udm_customer
SELECT ${columns:customer::c.%1$s AS %1$s}
FROM (

	SELECT ${columns:customer::s.%1$s AS %1$s}
	FROM udm_s_customer s
	WHERE lower(COALESCE(s.Source,'')) NOT LIKE '%james_avery_rental_derived%'

	UNION ALL

	SELECT ${columns:customer::u.%1$s AS %1$s}
	FROM (
		SELECT ${columns:customer:~SourceCustomerNumber,LastName,c_channelViewKeyCode,RowModified:dwc.%1$s AS %1$s}
			,dwc.SourceCustomerNumber AS SourceCustomerNumber
			,dwc.LastName AS LastName
			,sc.c_channelViewKeyCode AS c_channelViewKeyCode
			,unix_timestamp()*1000 AS RowModified
			,row_number() OVER (partition by dwc.SourceCustomerNumber order by dwc.forOrdering desc, dwc.RowModified desc) AS rwn
		FROM (
				SELECT 
					SourceCustomerNumber
					,c_channelViewKeyCode
					,LastName
				FROM udm_s_customer
				WHERE LOWER(COALESCE(Source,'')) LIKE '%james_avery_rental_derived%'
			) sc
			INNER JOIN (
				SELECT ${columns:customer::dw.%1$s AS %1$s}
					,substr(dw.SourceCustomerNumber,1,20) AS joinSCN,
					,dwc.LastName AS LastName
					,CASE 
						WHEN COALESCE(source,'') LIKE 'Customer%' THEN 3
						WHEN COALESCE(source,'') LIKE '%DMM%' THEN 2
						ELSE 1
					END AS forOrdering
				FROM udm_customer dw
				WHERE COALESCE(dw.LastName,'') <> ''
			) dwc ON sc.SourceCustomerNumber = dwc.joinSCN AND upper(sc.LastName) = upper(dwc.LastName)
		) u
	WHERE u.rwn = 1

	) c
;