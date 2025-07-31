SELECT assigned_jury_username, stadium_name
FROM MatchSession
WHERE rating = (SELECT MAX(rating)
				FROM MatchSession)
ORDER BY assigned_jury_username DESC;