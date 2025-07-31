SELECT J.name, J.surname, COUNT(*) as rated_sessions
FROM MatchSession M INNER JOIN Jury J ON M.assigned_jury_username = J.username
GROUP BY J.username
HAVING COUNT(*) >= ALL (SELECT COUNT(*) as rated_sessions
						FROM MatchSession M1
						GROUP BY M1.assigned_jury_username)