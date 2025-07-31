SELECT C.name, C.surname, COUNT(DISTINCT stadium_ID) AS directed_stadium_count
FROM MatchSession M INNER JOIN Team T ON M.team_ID = T.team_ID INNER JOIN Coach C ON T.coach_username = C.username
WHERE NOT EXISTS(
				SELECT *
				FROM MatchSession M1
                WHERE M1.stadium_ID NOT IN (SELECT M2.stadium_ID
								FROM MatchSession M2 INNER JOIN Team T2 ON M2.team_ID = T2.team_ID
                                WHERE T2.coach_username = C.username))
GROUP BY C.username