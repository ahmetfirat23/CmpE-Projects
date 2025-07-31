DELIMITER //

DROP TRIGGER IF EXISTS check_stadium; //

CREATE TRIGGER check_stadium
BEFORE INSERT ON MatchSession
FOR EACH ROW

BEGIN
    DECLARE m_count INT;
    
    SELECT COUNT(*) INTO m_count
    FROM MatchSession
    WHERE ((stadium_ID = NEW.stadium_ID) AND (stadium_name != NEW.stadium_name or stadium_country != NEW.stadium_country));
-- OR
 --   ((stadium_ID != NEW.stadium_ID) AND (stadium_name = NEW.stadium_name));
    
    IF m_count > 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Stadium integrity violation';
    END IF;
END; //