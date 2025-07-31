SELECT C.name, C.surname, T.channel_name, T.contract_start, T.contract_finish
FROM Coach C INNER JOIN Team T ON C.username = T.coach_username
WHERE T.channel_name = "Digiturk" AND
	STR_TO_DATE(T.contract_start, "%d.%m.%Y") <= "20240902" AND
    STR_TO_DATE(T.contract_finish, "%d.%m.%Y") >= "20251231"
ORDER BY C.name ASC;