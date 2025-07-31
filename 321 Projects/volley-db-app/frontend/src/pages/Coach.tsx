import { useNavigate, Navigate} from 'react-router-dom'
import { useAuth } from '../hooks'
import LogoutOutlinedIcon from '@mui/icons-material/LogoutOutlined';
import React, { useState, useEffect } from 'react';
import axios from 'axios';

import {Input} from '@/components/ui/input'
import { Button } from "@/components/ui/button"
import {
    Table,
    TableBody,
    TableCaption,
    TableCell,
    TableFooter,
    TableHead,
    TableHeader,
    TableRow,
  } from "@/components/ui/table"

import { format } from "date-fns"
import { Calendar as CalendarIcon } from "lucide-react"
 
import { cn } from "@/lib/utils"
import { Calendar } from "@/components/ui/calendar"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"

import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
  } from "@/components/ui/select"
import { parse } from 'path';
  

export default function Coach () {
    const navigate = useNavigate()
    const { logout, checkAuth, getAuth } = useAuth()
    const isAuth = checkAuth()

    const user = getAuth()

    const [positions, setPositions] = useState([])
    const [juries, setJuries] = useState([])
    

    useEffect(() => {
        axios.get(`http://localhost:8000/api/get-positions/`)
        .then(function (response) {
            setPositions(response.data.positions)
        })
        .catch(function (error) {
            console.log(error);
        });
        axios.get(`http://localhost:8000/api/get-juries/`)
        .then(function (response) {
            setJuries(response.data.juries)
        })
        .catch(function (error) {
            console.log(error);
        });
    }, [])

    if (!isAuth) return <Navigate to="/" />

    const [addResponseView, setAddResponseView] = React.useState({
        status: "",
        message: ""
    })
    const [deleteResponseView, setDeleteResponseView] = React.useState({
        status: "",
        message: ""
    })
    const [createResponseView, setCreateResponseView] = React.useState({
        status: "",
        message: ""
    })
    const [viewResponseView, setViewResponseView] = React.useState({
        status: "",
        message: ""
    })

    const [date, setDate] = React.useState<Date>()

    const [activeTab, setActiveTab] = React.useState('addMatchSession')

    const [sessions, setSessions] = React.useState([])
    const [players, setPlayers] = React.useState([])

    const [addTabState, setAddTabState] = React.useState({stadium_name: null, stadium_country: null, date: null, time_slot: null, assigned_jury_username: null})
    const [deleteTabState, setDeleteTabState] = React.useState(null)
    const [createTabState, setCreateTabState] = React.useState({ session_id: null, team_id: null, players: [null, null, null, null, null, null] })
    const [viewTabState, setViewTabState] = React.useState([])
 

    return (
        <div className='h-screen w-screen flex flex-col'>
            <div className='flex-initial flex flex-row justify-between bg-zinc-900 text-white py-3 px-5 items-center'>
                <h1 className='text-xl'>Coach</h1>
                <div className='flex flex-row bg-zinc-700 p-1 text-base gap-1 rounded-sm text-white whitespace-nowrap'>
                    <button className={activeTab === "addMatchSession" ? "bg-zinc-900 rounded-sm flex-1 py-1 px-4" : "flex-1 py-1 px-4 text-zinc-400"} onClick={() => setActiveTab('addMatchSession')}>Add Match Session</button>
                    <button className={activeTab === "deleteMatchSession" ? "bg-zinc-900 rounded-sm flex-1 py-1 px-4" : "flex-1 py-1 px-4 text-zinc-400"} onClick={() => setActiveTab('deleteMatchSession')}>Delete Match Session</button>
                    <button className={activeTab === "createSquad" ? "bg-zinc-900 rounded-sm flex-1 py-1 px-4" : "flex-1 py-1 px-4 text-zinc-400"} onClick={() => {
                        setActiveTab('createSquad')
                        axios.post(`http://localhost:8000/api/get-coach-sessions/`, {coach_username: user.user[0]})
                        .then(function (response) {
                            console.log(response.data.sessions);
                            setSessions(response.data.sessions);
                            setCreateTabState((prev) => {return { ...prev, session_id: null}});
                        })
                        .catch(function (error) {
                            console.log(error);
                            setCreateResponseView({status: "error", message: "An error occurred while fetching sessions!"});
                        });
                    }}>Create Squad</button>
                    <button className={activeTab === "viewStadiums" ? "bg-zinc-900 rounded-sm flex-1 py-1 px-4" : "flex-1 py-1 px-4 text-zinc-400"} onClick={() => {
                        setActiveTab('viewStadiums'); 
                        axios.get(`http://localhost:8000/api/get-stadiums/`)
                            .then(function (response) {
                                console.log(response.data.stadiums);
                                setViewTabState(response.data.stadiums);
                            })
                            .catch(function (error) {
                                console.log(error);
                                setViewResponseView({status: "error", message: "An error occurred while fetching stadiums!"});
                            });
                    }}>View Stadiums</button>
                </div>
                <button onClick={() => {logout(); navigate('/')}}><LogoutOutlinedIcon className='text-2xl'/></button>
            </div>
            <div className='h-full w-full flex-auto flex justify-center'>
                <div className='flex flex-col justify-center'>
                    <div className='shadow-sm rounded-md border p-7'>
                        {
                            activeTab === 'addMatchSession' && (
                                <div className='flex flex-col gap-2'>
                                    <h1 className="text-2xl font-bold">Add Match Session</h1>
                                    <div className="flex flex-row gap-2">
                                        <Input placeholder='Stadium Name' className='border' value={addTabState.stadium_name} onChange={(e) => setAddTabState((prev) => {return { ...prev, stadium_name: e.target.value}})}/>
                                        <Input placeholder='Stadium Country' className='border' value={addTabState.stadium_country} onChange={(e) => setAddTabState((prev) => {return { ...prev, stadium_country: e.target.value}})}/>
                                    </div>
                                    <div className="flex flex-row gap-2">
                                        <Popover>
                                            <PopoverTrigger asChild>
                                                <Button
                                                variant={"outline"}
                                                className={cn(
                                                    "w-full justify-start text-left font-normal",
                                                    !date && "text-muted-foreground"
                                                )}
                                                >
                                                <CalendarIcon className="mr-2 h-4 w-4" />
                                                {date ? format(date, "PPP") : <span>Date</span>}
                                                </Button>
                                            </PopoverTrigger>
                                            <PopoverContent className="w-auto p-0">
                                                <Calendar
                                                mode="single"
                                                selected={date}
                                                onSelect={
                                                    (date) => {
                                                        setDate(date)
                                                        setAddTabState((prev) => {return { ...prev, date: format(date, "dd.MM.yyyy")}})
                                                    }
                                                }
                                                initialFocus
                                                />
                                            </PopoverContent>
                                        </Popover>
                                        <Select onValueChange={
                                            (value) => {
                                                setAddTabState((prev) => {return { ...prev, time_slot: parseInt(value)}})
                                            }
                                        
                                        }>
                                            <SelectTrigger className="w-full">
                                                <SelectValue placeholder="Time Slot" />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="1">1</SelectItem>
                                                <SelectItem value="2">2</SelectItem>
                                                <SelectItem value="3">3</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                    <Select onValueChange={
                                        (value) => {
                                            setAddTabState((prev) => {return { ...prev, assigned_jury_username: value}})
                                        }
                                    
                                    }>
                                        <SelectTrigger className="w-full">
                                            <SelectValue placeholder="Assigned Jury" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {
                                                juries.map((jury) => (
                                                    <SelectItem value={jury[0]}>{jury[1] + " " + jury[2]}</SelectItem>
                                                ))
                                            }
                                        </SelectContent>
                                    </Select>
                                    <Button onClick={() => {
                                        axios.post(`http://localhost:8000/api/add-match-session/`, {
                                            ...addTabState,
                                            "coach_username": user.user[0],
                                        
                                        })
                                        .then(function (response) {
                                            setAddResponseView({status: "success", message: "Match session added successfully!"});
                                        })
                                        .catch(function (error) {
                                            console.log(error);
                                            setAddResponseView({status: "error", message: "An error occurred while adding match session!"});
                                        }); 
                                    }}>Add</Button>
                                    <div className={addResponseView.status === "" ? "hidden" : (addResponseView.status === "success" ? "text-green-500" : "text-red-500")}>
                                        {addResponseView.message}
                                    </div>
                                </div>
                            )
                        }
                        {
                            activeTab === 'deleteMatchSession' && (
                                <div className='flex flex-col gap-2'>
                                    <h1 className="text-2xl font-bold">Delete Match Session</h1>
                                    <Input placeholder='Session ID' className='' value={deleteTabState} onChange={(e) => setDeleteTabState(e.target.value)}/>
                                    <Button onClick={() => {
                                        axios.post(`http://localhost:8000/api/delete-match-session/`, {"session_ID": deleteTabState})
                                        .then(function (response) {
                                            setDeleteResponseView({status: "success", message: "Match session deleted successfully!"});
                                        })
                                        .catch(function (error) {
                                          console.log(error);
                                            setDeleteResponseView({status: "error", message: "An error occurred while deleting match session!"});
                                        }); 
                                    }}>Delete</Button>
                                    <div className={deleteResponseView.status === "" ? "hidden" : (deleteResponseView.status === "success" ? "text-green-500" : "text-red-500")}>
                                        {deleteResponseView.message}
                                    </div>
                                </div>
                            )
                        }
                        {
                            activeTab === 'createSquad' && (
                                <div className="flex flex-col gap-2">
                                    <h1 className="text-2xl font-bold">Create Squad</h1>
                                    <div className="flex flex-col gap-2">
                                        <Select onValueChange={
                                            (value) => {
                                                setCreateTabState((prev) => {return { ...prev, session_id: value[0], team_id: value[1]}})
                                                axios.post(`http://localhost:8000/api/get-team-players/`, {team_id: value[1]})
                                                .then(function (response) {
                                                    setPlayers(response.data.players);
                                                    console.log(response.data.players);
                                                })
                                                .catch(function (error) {
                                                    console.log(error);
                                                    setCreateResponseView({status: "error", message: "An error occurred while fetching players!"});
                                                });
                                            }
                                        
                                        }>
                                            <SelectTrigger className="w-full">
                                                <SelectValue placeholder="Select Session" />
                                            </SelectTrigger>
                                            <SelectContent>
                                                {
                                                    sessions.map((session) => (
                                                        <SelectItem value={session}>{session[0]}</SelectItem>
                                                    ))
                                                }
                                            </SelectContent>
                                        </Select>
                                        {
                                            [0,1,2,3,4,5].map((i) => (
                                                <div key={i} className='flex flex-row gap-2'>
                                                    <Select onValueChange={
                                                        (value) => {
                                                            setCreateTabState((prev) => {return { ...prev, players: prev.players.map((player, index) => index === i ? {username: value, position: player ? player.position : null} : player)}});
                                                        }
                                                    
                                                    } disabled={!createTabState.session_id}>
                                                        <SelectTrigger className="w-full">
                                                            <SelectValue placeholder="Select Player" />
                                                        </SelectTrigger>
                                                        <SelectContent>
                                                            {
                                                                players.map((player) => (
                                                                        player
                                                                        &&
                                                                        <SelectItem value={player.username}>{player.name + " " + player.surname}</SelectItem>
                                                                ))
                                                            }
                                                        </SelectContent>
                                                    </Select>
                                                    <Select onValueChange={
                                                        (value) => {
                                                            setCreateTabState((prev) => {return { ...prev, players: prev.players.map((player, index) => index === i ? {username: player ? player.username : null, position: value} : player)}});
                                                        }
                                                    
                                                    } disabled={!createTabState.players[i]}>
                                                        <SelectTrigger className="w-full">
                                                            <SelectValue placeholder="Select Position" />
                                                        </SelectTrigger>
                                                        <SelectContent>
                                                            {
                                                                createTabState.players[i]
                                                                &&
                                                                players.filter((player) => player.username === createTabState.players[i].username)[0].positions.map((position) => (
                                                                    <SelectItem value={position}>{positions.filter((p) => p[0] == position)[0][1]}</SelectItem>
                                                                ))
                                                            }
                                                        </SelectContent>
                                                    </Select>
                                                </div>
                                            ))
                                        }
                                    </div>
                                    <Button onClick={() => {
                                        axios.post(`http://localhost:8000/api/create-squad/`, {session_id: createTabState.session_id, players: createTabState.players, coach_username: user.user[0]})
                                        .then(function (response) {
                                            setCreateResponseView({status: "success", message: "Squad created successfully!"});
                                        })
                                        .catch(function (error) {
                                          console.log(error);
                                            setCreateResponseView({status: "error", message: "An error occurred while creating squad!"});
                                        }); 
                                    }}>Create</Button>
                                    <div className={createResponseView.status === "" ? "hidden" : (createResponseView.status === "success" ? "text-green-500" : "text-red-500")}>
                                        {createResponseView.message}
                                    </div>
                                </div>
                            )
                        }
                        {
                            activeTab === 'viewStadiums' && (
                                <div>
                                    <div>
                                        <Table>
                                            <TableHeader>
                                                <TableRow>
                                                <TableHead className="w-[100px]">Stadium</TableHead>
                                                <TableHead className="text-right">Country</TableHead>
                                                </TableRow>
                                            </TableHeader>
                                            <TableBody>
                                                {viewTabState.map((stadium) => (
                                                <TableRow key={stadium[0]}>
                                                    <TableCell className="font-medium whitespace-nowrap">{stadium[0]}</TableCell>
                                                    <TableCell className="text-right">{stadium[1]}</TableCell>
                                                </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </div>
                                    <div className={viewResponseView.status === "" ? "hidden" : "text-red-500"}>
                                        {viewResponseView.message}
                                    </div>
                                </div>
                            )
                        }
                    </div>
                </div>
            </div>
        </div>
    )
}