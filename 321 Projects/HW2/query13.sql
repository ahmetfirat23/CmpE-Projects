SELECT M.stadium_name, C.name, C.surname, COUNT(*) as directed_count
FROM MatchSession M INNER JOIN Team T ON M.team_ID = T.team_ID INNER JOIN Coach C ON T.coach_username = C.username
GROUP BY M.stadium_name, C.username
HAVING COUNT(*) >= ALL(SELECT COUNT(*)
						FROM MatchSession M1 INNER JOIN Team T1 ON M1.team_ID = T1.team_ID INNER JOIN Coach C1 ON T1.coach_username = C1.username
						GROUP BY M1.stadium_name, C1.username
                        HAVING M.stadium_name = M1.stadium_name)
