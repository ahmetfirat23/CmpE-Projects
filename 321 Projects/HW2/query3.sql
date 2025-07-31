SELECT *
FROM MatchSession
WHERE rating = (SELECT MIN(rating)
				FROM MatchSession);