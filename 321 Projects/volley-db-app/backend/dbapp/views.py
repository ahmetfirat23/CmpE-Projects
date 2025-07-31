from django.shortcuts import render, get_object_or_404
from django.db import connection
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from rest_framework import status
from datetime import datetime

from .serializers import UserSerializer

@api_view(['GET'])
def get_positions(request):
    cursor = connection.cursor()
    cursor.execute("SELECT position_id, position_name FROM Position")
    positions = cursor.fetchall()
    return Response({'positions': positions})


@api_view(['GET'])
def get_teams(request):
    cursor = connection.cursor()
    cursor.execute("SELECT team_id, team_name FROM Team")
    teams = cursor.fetchall()
    return Response({'teams': teams})

@api_view(['GET'])
def get_juries(request):
    cursor = connection.cursor()
    cursor.execute("SELECT username, name, surname FROM Jury")
    juries = cursor.fetchall()
    return Response({'juries': juries})


@api_view(['POST'])
def get_coach_sessions(request):
    data = request.data
    coach_username = data['coach_username']
    cursor = connection.cursor()
    cursor.execute(f'SELECT session_ID, team_ID FROM MatchSession WHERE team_ID IN (SELECT team_ID FROM Team WHERE coach_username = "{coach_username}") AND session_ID NOT IN (SELECT session_ID FROM SessionSquads)')
    sessions = cursor.fetchall()
    return Response({'sessions': sessions})


@api_view(['POST'])
def get_jury_sessions(request):
    data = request.data
    jury_username = data['jury_username']
    cursor = connection.cursor()
    cursor.execute(f'SELECT session_ID FROM MatchSession WHERE assigned_jury_username = "{jury_username}" AND rating IS NULL')
    sessions = cursor.fetchall()
    return Response({'sessions': sessions})

@api_view(['POST'])
def get_team_players(request):
    data = request.data
    team_id = data['team_id']
    cursor = connection.cursor()
    cursor.execute(f'SELECT username, name, surname FROM Player WHERE username IN (SELECT username FROM PlayerTeams WHERE team = {team_id})')
    player_list = cursor.fetchall()
    players = []
    for player in player_list:
        cursor.execute(f'SELECT position FROM PlayerPositions WHERE username = "{player[0]}"')
        positions = cursor.fetchall()
        players.append({'username': player[0], 'name': player[1], 'surname': player[2], 'positions': [position[0] for position in positions]})
    return Response({'players': players})


@api_view(['POST'])
def create_user(request):
    data = request.data
    usertype = data['usertype']        
    username = data['username']
    password = data['password']
    name = data['name']
    surname = data['surname']
    if username == "" or password == "" or name == "" or surname == "":
        return Response({'error': 'Empty fields'}, status=status.HTTP_400_BAD_REQUEST)
    cursor = connection.cursor()
    for user_type in ['DBManager', 'Player', 'Coach', 'Jury']:
        user = cursor.execute("SELECT * FROM " + user_type + " WHERE username = %s", [username])
        if user:
            return Response({'error': 'Username already exists'}, status=status.HTTP_409_CONFLICT)

    if usertype == "Player":
        date_of_birth = data['date_of_birth']
        height = data['height']
        weight = data['weight']
        team_ids = data['team_ids']
        position_ids = data['position_ids']

        if date_of_birth == "" or height == "" or weight == "" or team_ids == [] or position_ids == []:
            return Response({'error': 'Empty fields'}, status=status.HTTP_400_BAD_REQUEST)
        
        cursor.execute("INSERT INTO Player (username, password, name, surname, date_of_birth, height, weight) VALUES (%s, %s, %s, %s, %s, %s, %s)", [username, password, name, surname, date_of_birth, height, weight])
        for team_id in set(team_ids):
            cursor.execute("SELECT MAX(player_teams_id) FROM PlayerTeams")
            player_teams_id = cursor.fetchone()[0] + 1
            cursor.execute(f'INSERT INTO PlayerTeams (player_teams_id, username, team) VALUES ({player_teams_id}, "{username}", {team_id})')
        for position_id in set(position_ids):
            cursor.execute("SELECT MAX(player_positions_id) FROM PlayerPositions")
            player_positions_id = cursor.fetchone()[0] + 1
            cursor.execute(f'INSERT INTO PlayerPositions (player_positions_id, username, position) VALUES ({player_positions_id}, "{username}", {position_id})')

    elif usertype == "Coach":
        nationality = data['nationality']
        if nationality == "":
            return Response({'error': 'Empty fields'}, status=status.HTTP_400_BAD_REQUEST)

        cursor.execute("INSERT INTO Coach (username, password, name, surname,  nationality) VALUES (%s, %s, %s, %s, %s)", [username, password, name, surname, nationality])
        
    elif usertype == "Jury":
        nationality = data['nationality']
        if nationality == "":
            return Response({'error': 'Empty fields'}, status=status.HTTP_400_BAD_REQUEST)
        cursor.execute("INSERT INTO Jury (username, password, name, surname, nationality) VALUES (%s, %s, %s, %s, %s)", [username, password, name, surname, nationality])
    else:
        return Response({'error': 'Invalid user type'}, status=status.HTTP_400_BAD_REQUEST)
    
    return Response(request.data, status=status.HTTP_200_OK)


@api_view(['POST'])
def login(request):
    cursor = connection.cursor()
    types = ['DBManager', 'Player', 'Coach', 'Jury']
    for user_type in types:
        user = cursor.execute("SELECT * FROM " + user_type + " WHERE username = %s AND password = %s", [request.data['username'], request.data['password']])
        if user:
            break
    if not user:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
    
    user = cursor.fetchone()
    return Response({'user': user, "type": user_type})

@api_view(['POST'])
def update_stadium(request):
    data = request.data
    previous_id = data['previous_id']
    name = data['name']
    if not name or (not previous_id and previous_id != 0):
        return Response("Empty fields", status=status.HTTP_400_BAD_REQUEST)
    cursor = connection.cursor()
    cursor.execute("UPDATE MatchSession SET stadium_name = %s WHERE stadium_ID = %s", [name, previous_id])
    print(cursor.rowcount)
    return Response("Stadium not found" if cursor.rowcount == 0 else f"{cursor.rowcount} stadium names changed.", status=status.HTTP_200_OK)
    
@api_view(['GET'])
def get_stadiums(request):
    cursor = connection.cursor()
    cursor.execute("SELECT DISTINCT stadium_name, stadium_country, stadium_ID FROM MatchSession")
    stadiums = cursor.fetchall()
    return Response({'stadiums': stadiums})

@api_view(['POST'])
def delete_match_session(request):
    cursor = connection.cursor()
    try:
        session_ID = request.data['session_ID']
        if not session_ID and session_ID != 0:
            return Response("No session ID provided", status=status.HTTP_400_BAD_REQUEST)
        cursor.execute("DELETE FROM MatchSession WHERE session_ID = %s", [session_ID])
        cursor.execute("DELETE FROM SessionSquads WHERE session_ID = %s", [session_ID])
        return Response("Match session deleted", status=status.HTTP_200_OK)
    except:
        return Response("Match session not found", status=status.HTTP_404_NOT_FOUND)

@api_view(['POST'])
def add_match_session(request):
    data = request.data
    coach_username = data['coach_username']
    stadium_name = data['stadium_name']
    stadium_country = data['stadium_country']
    date = data['date']
    time_slot = data['time_slot']
    assigned_jury_username = data['assigned_jury_username']
    if not assigned_jury_username:
        return Response("No jury assigned", status=status.HTTP_400_BAD_REQUEST)
    cursor = connection.cursor()
    cursor.execute("SELECT MAX(session_ID) FROM MatchSession")
    session_id = cursor.fetchone()[0] + 1
    cursor.execute("SELECT DISTINCT stadium_ID FROM MatchSession WHERE stadium_name = %s AND stadium_country = %s" , [stadium_name, stadium_country])
    stadium = cursor.fetchone()
    if not stadium:
        return Response("Stadium not found", status=status.HTTP_404_NOT_FOUND)
    else:
        stadium_id = stadium[0]
    cursor.execute(f'SELECT team_ID FROM Team WHERE coach_username = "{coach_username}" AND STR_TO_DATE(contract_start, "%d.%m.%Y") <= STR_TO_DATE("{date}", "%d.%m.%Y") AND STR_TO_DATE(contract_finish, "%d.%m.%Y") > STR_TO_DATE("{date}", "%d.%m.%Y")')
    team_id = cursor.fetchone()
    if not team_id:
        return Response("Coach team mismatch", status=status.HTTP_400_BAD_REQUEST)
    team_id = team_id[0]
    cursor.execute(f'INSERT INTO MatchSession (session_ID, team_ID, stadium_ID, stadium_name, stadium_country, time_slot, date, assigned_jury_username, rating) VALUES ({session_id}, {team_id}, {stadium_id}, "{stadium_name}", "{stadium_country}", {time_slot}, "{date}", "{assigned_jury_username}", NULL)')
    return Response("Match session added", status=status.HTTP_200_OK)

@api_view(['POST'])
def create_squad(request):
    data = request.data
    print(data)
    coach_username = data['coach_username']
    if not coach_username:
        return Response("No coach username provided", status=status.HTTP_400_BAD_REQUEST)
    session_id = data['session_id']
    if not session_id and session_id != 0:
        return Response("No session ID provided", status=status.HTTP_400_BAD_REQUEST)
    players = data['players']
    if not players:
        return Response("No players in squad", status=status.HTTP_400_BAD_REQUEST)
    if len(players) != 6:
        return Response("Squad must have 6 players", status=status.HTTP_400_BAD_REQUEST)
    cursor = connection.cursor()
    cursor.execute(f'SELECT * FROM MatchSession WHERE session_ID = {session_id}')
    session = cursor.fetchone()
    if not session:
        return Response("Session not found", status=status.HTTP_404_NOT_FOUND)
    team_id = session[1]
    date = session[6]
    time_slot = session[5]
    cursor.execute(f'SELECT * FROM Team WHERE team_ID = {team_id} AND coach_username = "{coach_username}" AND STR_TO_DATE(contract_start, "%d.%m.%Y") <= STR_TO_DATE("{date}", "%d.%m.%Y") AND STR_TO_DATE(contract_finish, "%d.%m.%Y") > STR_TO_DATE("{date}", "%d.%m.%Y")')
    if not cursor.fetchone():
        return Response("Coach team mismatch", status=status.HTTP_400_BAD_REQUEST)
    cursor.execute(f'SELECT * FROM SessionSquads WHERE session_ID = {session_id}')
    if cursor.fetchone():
        return Response("Squad already exists", status=status.HTTP_400_BAD_REQUEST)
    
    player_usernames = [player['username'] for player in players]
    if len(player_usernames) != len(set(player_usernames)):
        return Response("Duplicate players in squad", status=status.HTTP_400_BAD_REQUEST)
    
    for player in players:
        cursor.execute(f'SELECT * FROM Player WHERE username= "{player["username"].strip()}"')
        player_data = cursor.fetchone()
        if not player_data:
            print(player)
            return Response("Player not found", status=status.HTTP_404_NOT_FOUND)
        cursor.execute(f'SELECT * FROM PlayerTeams WHERE username = "{player_data[0]}" AND team = {team_id}')
        if not cursor.fetchone():
            return Response("Player not in team", status=status.HTTP_400_BAD_REQUEST)
        cursor.execute(f'SELECT * FROM PlayerPositions WHERE username = "{player_data[0]}" AND position = {player["position"]}')
        if not cursor.fetchone():
            return Response("Player position mismatch", status=status.HTTP_400_BAD_REQUEST)
        cursor.execute(f'SELECT * FROM MatchSession WHERE STR_TO_DATE(date, "%d.%m.%Y") = STR_TO_DATE("{date}", "%d.%m.%Y") AND (time_slot = {time_slot} OR time_slot = {time_slot} + 1 OR time_slot = {time_slot} - 1) AND session_ID IN (SELECT session_ID FROM SessionSquads WHERE played_player_username = "{player_data[0]}")')
        if cursor.fetchone():
            return Response("Player already played in a session", status=status.HTTP_400_BAD_REQUEST)
    for player in players:
        cursor.execute(f'SELECT username FROM Player WHERE username = "{player["username"].strip()}"')
        username = cursor.fetchone()[0]
        cursor.execute(f'SELECT MAX(squad_ID) FROM SessionSquads')
        squad_id = cursor.fetchone()[0] + 1
        cursor.execute(f'INSERT INTO SessionSquads (squad_ID, session_ID, played_player_username, position_ID) VALUES ({squad_id}, {session_id}, "{username}", {player["position"]})')
    return Response("Squad created", status=status.HTTP_200_OK)


@api_view(['POST'])
def view_rating_stats(request):
    data = request.data
    jury_username = data['jury_username']
    cursor = connection.cursor()
    cursor.execute(f'SELECT AVG(rating), COUNT(rating) FROM MatchSession WHERE assigned_jury_username = "{jury_username}" AND rating IS NOT NULL')
    rating, count = cursor.fetchone()
    if count == 0:
        return Response("No ratings found", status=status.HTTP_404_NOT_FOUND)
    return Response({'average_rating': rating, "rating_count": count})


@api_view(["POST"])
def rate_match_session(request):
    data = request.data
    session_id = data['session_id']
    if not session_id and session_id != 0:
        return Response("No session ID provided", status=status.HTTP_400_BAD_REQUEST)
    rating = data['rating']
    if (not rating and rating != 0):
        return Response("No rating provided", status=status.HTTP_400_BAD_REQUEST)
    if rating < 0 or rating > 5:
        return Response("Rating must be between 0 and 5", status=status.HTTP_400_BAD_REQUEST)
    jury_username = data['jury_username']
    cursor = connection.cursor()
    cursor.execute(f'SELECT * FROM MatchSession WHERE session_ID = {session_id}')
    session = cursor.fetchone()
    if not session:
        return Response("Match session not found", status=status.HTTP_404_NOT_FOUND)
    date, assigned_jury_username, rated = session[6], session[7], session[8]
    if datetime.strptime(date,f"%d.%m.%Y") > datetime.now():
        return Response("Match session not yet played", status=status.HTTP_400_BAD_REQUEST)
    if assigned_jury_username != jury_username:
        return Response("Unauthorized jury", status=status.HTTP_401_UNAUTHORIZED)
    if rated:
        return Response("Match session already rated", status=status.HTTP_400_BAD_REQUEST)
    cursor.execute(f'UPDATE MatchSession SET rating = {rating} WHERE session_ID = {session_id}')
    return Response("Match session rated", status=status.HTTP_200_OK)


@api_view(['POST'])
def view_players(request):
    data = request.data
    player_username = data['player_username']
    cursor = connection.cursor()
    cursor.execute(f"""SELECT name, surname, height, COUNT(*) 
                   FROM SessionSquads S 
                   INNER JOIN Player P ON S.played_player_username = P.username 
                   WHERE played_player_username != "{player_username}" AND session_ID IN 
                        (SELECT session_ID 
                        FROM SessionSquads 
                        WHERE played_player_username = "{player_username}")
                   GROUP BY username""")
    players = cursor.fetchall()
    cursor.execute(f"""
        SELECT AVG(height) FROM
            (SELECT height FROM SessionSquads S 
            INNER JOIN Player P ON S.played_player_username = P.username
            WHERE played_player_username != "{player_username}" 
            AND session_ID IN 
                (SELECT session_ID FROM SessionSquads
		        WHERE played_player_username = "{player_username}")
	            GROUP BY username
	            HAVING COUNT(*) >= ALL 
                    (SELECT COUNT(*) FROM SessionSquads S INNER JOIN Player P ON S.played_player_username = P.username
		            WHERE played_player_username != "{player_username}" AND session_ID IN 
                        (SELECT session_ID FROM SessionSquads
                        WHERE played_player_username = "{player_username}")
		            GROUP BY username)) AS H""")
    avg_height = cursor.fetchone()[0]
    return Response({"avg_height": avg_height, "players": sorted(players, key=lambda x: x[3], reverse=True)})


@api_view(['GET'])
def get_match_sessions(request):
    cursor = connection.cursor()
    cursor.execute("SELECT session_id FROM MatchSession")
    session_ids = cursor.fetchall()
    return Response({'session_ids': session_ids})


@api_view(['GET'])
def get_players(request):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM Player")
    players = cursor.fetchall()
    return Response({'players': players})