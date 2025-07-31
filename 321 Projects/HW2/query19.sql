SELECT P.name, P.surname 
FROM MatchSession M INNER JOIN SessionSquads SS ON M.session_ID = SS.session_ID
	INNER JOIN Player P ON SS.played_player_username = P.username 
    INNER JOIN Position Pos ON SS.position_ID = Pos.position_ID
WHERE TRIM(Pos.position_name) = "Libero" AND TRIM(M.stadium_name) = "GD Voleybol Arena"
GROUP BY P.username;
