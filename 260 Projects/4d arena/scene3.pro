%% CONSTANTS %%
height(10).
width(10).
universe_limit(9).
%% GLOBAL VARIABLES %%
global_state_id(0).
global_universe_id(0).

% state(StateId, Agents, CurrentTurn, TurnOrder).
% Agents = agent_dict{AgentId: agent{X, Y, Class}, ...}
% CurrentTurn = TurnIndex
% TurnOrder = [AgentId, AgentId, ...]
state(0, agent_dict{0: agent{x: 1, y: 1, class: warrior, health: 100, mana: 100, agility: 2, armor: 8, name: 'Arthur'},
                    1: agent{x: 3, y: 4, class: wizard, health: 100, mana: 100, agility: 5, armor: 2, name: 'Merlin'},
                    2: agent{x: 4, y: 5, class: rogue, health: 100, mana: 100, agility: 8, armor: 5, name: 'Mordred'},
                    3: agent{x: 1, y: 6, class: rogue, health: 100, mana: 100, agility: 8, armor: 5, name: 'Illidan'},
                    4: agent{x: 6, y: 1, class: warrior, health: 100, mana: 100, agility: 2, armor: 8, name: 'Arthas'},
                    5: agent{x: 7, y: 7, class: wizard, health: 100, mana: 100, agility: 5, armor: 2, name: 'Harry'}
                    },
                    0,
                    [0, 1, 2, 3, 4, 5]).
% history(StateId, UniverseId, Time, Turn).
history(0, 0, 0, 0).

% current_time(UniverseId, Time, Turn).
current_time(0, 0, 0).