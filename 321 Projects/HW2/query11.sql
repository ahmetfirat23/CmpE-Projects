SELECT M.session_ID, C.name, C.surname, M.stadium_name, M.stadium_country, T.team_name
FROM MatchSession M INNER JOIN Team T ON M.team_ID = T.team_ID INNER JOIN Coach C ON T.coach_username = C.username
WHERE TRIM(C.name) = "Daniele" AND TRIM(C.surname) = "Santarelli" AND TRIM(M.stadium_country) <> "UK"
ORDER BY M.session_ID ASC;