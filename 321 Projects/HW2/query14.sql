SELECT M.session_ID, C.name, C.surname
FROM MatchSession M INNER JOIN Team T ON M.team_ID = T.team_ID INNER JOIN Coach C ON T.coach_username = C.username
WHERE TRIM(C.name) <> "Ferhat" AND TRIM(C.surname) <> "Akbaş" AND
	STR_TO_DATE(M.date, "%d.%m.%Y") >= "20240101"
ORDER BY M.session_ID ASC;