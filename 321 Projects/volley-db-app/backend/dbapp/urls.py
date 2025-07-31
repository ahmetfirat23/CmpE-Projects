from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login, name='login'),
    path('create-user/', views.create_user, name='create-user'),
    path('get-positions/', views.get_positions, name='get-positions'),
    path('get-teams/', views.get_teams, name='get-teams'),
    path('get-juries/', views.get_juries, name='get-juries'),
    path('get-coach-sessions/', views.get_coach_sessions, name='get-coach-sessions'),
    path('get-jury-sessions/', views.get_jury_sessions, name='get-jury-sessions'),
    path('get-team-players/', views.get_team_players, name='get-team-players'),
    path('get-stadiums/', views.get_stadiums, name='get-stadiums'),
    path('get-players/', views.get_players, name='get-players'),
    path('update-stadium/', views.update_stadium, name='update-stadium'),
    path('get-stadiums/', views.get_stadiums, name='get-stadiums'),
    path('delete-match-session/', views.delete_match_session, name='delete-match-session'),
    path('add-match-session/', views.add_match_session, name='add-match-session'),
    path("get-match-sessions/", views.get_match_sessions, name="get-match-sessions"),   
    path('create-squad/', views.create_squad, name='create-squad'),
    path('view-rating-stats/', views.view_rating_stats, name='view-rating-stats'),
    path('rate-match-session/', views.rate_match_session, name='rate-match-session'),
    path('view-players/', views.view_players, name='view-players'),
]