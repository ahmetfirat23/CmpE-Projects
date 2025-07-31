SELECT DISTINCT P.name, P.surname, IF(COUNT(DISTINCT PP.position)>1, "TRUE", "FALSE") AS more_than_one
FROM Player P INNER JOIN PlayerPositions PP ON P.username = PP.username 
GROUP BY P.username
ORDER BY P.name ASC