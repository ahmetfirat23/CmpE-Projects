SELECT T.team_name, Q.name, Q.surname, Q.played_count
FROM (SELECT DISTINCT T.team_name
	FROM Team T) AS T 
    
	LEFT JOIN
    
    (SELECT T.team_name, C.name, C.surname, DATEDIFF(STR_TO_DATE(T.contract_finish, "%d.%m.%Y"), STR_TO_DATE(T.contract_start, "%d.%m.%Y")) as played_count,  M.rating
	FROM MatchSession M RIGHT JOIN Team T ON M.team_ID = T.team_ID INNER JOIN Coach C ON T.coach_username = C.username
	WHERE C.username NOT IN ( 
		SELECT T1.coach_username
    	FROM MatchSession M1 RIGHT JOIN Team T1 ON M1.team_ID = T1.team_ID
		WHERE 4.7 > ANY (SELECT M2.rating
						FROM MatchSession M2 RIGHT JOIN Team T2 ON M2.team_ID = T2.team_ID
                        WHERE T1.coach_username = T2.coach_username))
	AND 
    DATEDIFF(STR_TO_DATE(T.contract_finish, "%d.%m.%Y"), STR_TO_DATE(T.contract_start, "%d.%m.%Y")) >= ALL 
		(SELECT DATEDIFF(STR_TO_DATE(T1.contract_finish, "%d.%m.%Y"), STR_TO_DATE(T1.contract_start, "%d.%m.%Y"))
			FROM Team T1 INNER JOIN Coach C1 ON T1.coach_username = C1.username
			WHERE T1.team_ID = T.team_ID)) AS Q
            
	ON T.team_name = Q.team_name
