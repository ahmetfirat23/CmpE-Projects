DELIMITER //

DROP TRIGGER IF EXISTS rating_change_limit; //

CREATE TRIGGER rating_change_limit
BEFORE UPDATE ON MatchSession
FOR EACH ROW
BEGIN
    DECLARE m_count INT;
    SELECT COUNT(*) INTO m_count
    FROM MatchSession
    WHERE rating IS NOT NULL AND rating != NEW.rating AND session_ID = NEW.session_ID;

    IF m_count > 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Rating already set!';
    END IF;
END; //
