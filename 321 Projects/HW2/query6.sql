SELECT T.team_name, C.name, C.surname, COUNT(PT.username) as player_count
FROM Team AS T LEFT JOIN PlayerTeams AS PT ON T.team_id = PT.team INNER JOIN Coach AS C ON T.coach_username = C.username
GROUP BY T.team_id