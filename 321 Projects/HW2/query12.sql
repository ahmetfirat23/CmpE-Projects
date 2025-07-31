SELECT C.name, C.surname,YEAR(STR_TO_DATE(M.date, "%d.%m.%Y")) as year, AVG(M.rating) 
FROM MatchSession M INNER JOIN Team T ON M.team_ID = T.team_ID INNER JOIN Coach C ON T.coach_username = C.username
GROUP BY YEAR(STR_TO_DATE(M.date, "%d.%m.%Y")), M.team_ID
HAVING AVG(M.rating) >= ALL (
	SELECT average
    FROM(
		SELECT AVG(M1.rating) as average , YEAR(STR_TO_DATE(M1.date, "%d.%m.%Y")) as year1
		FROM MatchSession M1 INNER JOIN Team T1 ON M1.team_ID = T1.team_ID
		GROUP BY YEAR(STR_TO_DATE(M1.date, "%d.%m.%Y")), M1.team_ID
		HAVING year = year1) as subquery)
ORDER BY year ASC