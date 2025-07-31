DELIMITER //

DROP TRIGGER IF EXISTS team_existance_trigger; //

CREATE TRIGGER team_existance_trigger
BEFORE INSERT ON PlayerTeams
FOR EACH ROW
BEGIN
    DECLARE m_count INT;
    SELECT COUNT(*) INTO m_count
    FROM Team
    WHERE team_ID = NEW.team;

    IF m_count = 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Team does not exist!';
    END IF;
END; //