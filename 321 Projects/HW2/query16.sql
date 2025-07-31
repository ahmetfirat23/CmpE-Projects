SELECT C.name, C.surname, AVG(rating) as average_rating
FROM MatchSession M INNER JOIN Team T ON M.team_ID = T.team_ID INNER JOIN Coach C ON T.coach_username = C.username
GROUP BY C.username
ORDER BY C.name DESC;