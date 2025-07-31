import { Button } from "@/components/ui/button"
import { useNavigate, Navigate} from 'react-router-dom'
import { useAuth } from '../hooks'
import LogoutOutlinedIcon from '@mui/icons-material/LogoutOutlined';
import React, { useEffect, useState } from 'react';
import axios from 'axios';

import {Input} from '@/components/ui/input'

import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
  } from "@/components/ui/dropdown-menu"

import { format, set } from "date-fns"
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
  

export default function DBManager () {
    const navigate = useNavigate()
    const { logout, checkAuth, getAuth } = useAuth()
    const isAuth = checkAuth()

    const [positions, setPositions] = useState([])
    const [teams, setTeams] = useState([])
    const [stadiums, setStadiums] = useState([])
    const [date, setDate] = React.useState<Date>()

    const [createData, setCreateData] = React.useState({
        username: null,
        password: null,
        name: null,
        surname: null,
        date_of_birth: null,
        height: null,
        weight: null,
        team_ids: [],
        position_ids: [],
        nationality: null,
    })

    useEffect(() => {
        axios.get(`http://localhost:8000/api/get-positions/`)
        .then(function (response) {
            setPositions(response.data.positions)
        })
        .catch(function (error) {
            console.log(error);
        });
        axios.get(`http://localhost:8000/api/get-teams/`)
        .then(function (response) {
            setTeams(response.data.teams)
        })
        .catch(function (error) {
            console.log(error);
        });
        axios.get(`http://localhost:8000/api/get-stadiums/`)
        .then(function (response) {
            setStadiums(response.data.stadiums)
            console.log(response.data.stadiums)
        })
        .catch(function (error) {
            console.log(error);
        });
    }, [])

    if (!isAuth) return <Navigate to="/" />

    const user = getAuth()

    const [createResponseView, setCreateResponseView] = React.useState({
        status: "",
        message: ""
    })
    const [updateResponseView, setUpdateResponseView] = React.useState({
        status: "",
        message: ""
    })
    
    const [activeTab, setActiveTab] = React.useState('Coach')
    const [updateData, setUpdateData] = React.useState({
        previous_id: null,
        name: null
    })
    const [teamInputValue, setTeamInputValue] = React.useState("")
    const [positionInputValue, setPositionInputValue] = React.useState("")

    return (
        <div className='h-screen w-screen flex flex-col'>
            <div className='flex-initial flex flex-row justify-between bg-zinc-900 text-white py-3 px-5'>
                <h1 className='text-xl'>DBManager</h1>
                <button onClick={() => {logout(); navigate('/')}}><LogoutOutlinedIcon className='text-2xl'/></button>
            </div>
            <div className="p-8">
                <div className="flex flex-row gap-5">
                    <div className="flex-[70%] rounded-md shadow-sm border p-5 flex flex-col gap-2">
                        <h1 className="text-2xl font-bold">Create User</h1>
                        <div className='flex flex-row bg-zinc-100 p-1 text-base gap-1 rounded-sm text-zinc-800'>
                            <button className={activeTab === "Coach" ? "bg-white rounded-sm flex-1 p-1" : "flex-1 p-1 text-zinc-500"} onClick={() => setActiveTab('Coach')}>Coach</button>
                            <button className={activeTab === "Jury" ? "bg-white rounded-sm flex-1 p-1" : "flex-1 p-1 text-zinc-500"} onClick={() => setActiveTab('Jury')}>Jury</button>
                            <button className={activeTab === "Player" ? "bg-white rounded-sm flex-1 p-1" : "flex-1 p-1 text-zinc-500"} onClick={() => setActiveTab('Player')}>Player</button>
                        </div>
                        <div className="flex flex-row gap-2">
                            <Input placeholder="Username" value={createData.username} onChange={(e) => setCreateData((prev) => {return { ...prev, username: e.target.value}})}/>
                            <Input placeholder="Password" type='password' value={createData.password} onChange={(e) => setCreateData((prev) => {return { ...prev, password: e.target.value}})}/>
                        </div>
                        <div className="flex flex-row gap-2">
                            <Input placeholder="Name" value={createData.name} onChange={(e) => setCreateData((prev) => {return { ...prev, name: e.target.value}})}/>
                            <Input placeholder="Surname" value={createData.surname} onChange={(e) => setCreateData((prev) => {return { ...prev, surname: e.target.value}})}/>
                        </div>
                        {
                            (activeTab === "Coach" || activeTab === "Jury") && (
                                <Input placeholder="Nationality" value={createData.nationality} onChange={(e) => setCreateData((prev) => {return { ...prev, nationality: e.target.value}})}/>
                            )
                        }
                        {
                            activeTab === "Player" && (
                                <>
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
                                            {date ? format(date, "PPP") : <span>Date of Birth</span>}
                                            </Button>
                                        </PopoverTrigger>
                                        <PopoverContent className="w-auto p-0">
                                            <Calendar
                                            mode="single"
                                            selected={date}
                                            onSelect={
                                                (date) => {
                                                    setDate(date)
                                                    setCreateData((prev) => {return { ...prev, date_of_birth: format(date, "dd/MM/yyyy")}})
                                                }
                                            }
                                            initialFocus
                                            />
                                        </PopoverContent>
                                    </Popover>
                                    <div className="flex flex-row gap-2">
                                        <Input placeholder="Height" value={createData.height} onChange={(e) => setCreateData((prev) => {return { ...prev, height: e.target.value}})}/>
                                        <Input placeholder="Weight" value={createData.weight} onChange={(e) => setCreateData((prev) => {return { ...prev, weight: e.target.value}})}/>
                                    </div>
                                    <div className="flex flex-row gap-2">
                                        <div className="w-full">
                                            <DropdownMenu>
                                                <DropdownMenuTrigger className="w-full">
                                                    <Button className="w-full" variant="outline">Teams</Button>
                                                </DropdownMenuTrigger>
                                                <DropdownMenuContent>
                                                    {
                                                        teams.map((team) => (
                                                            <DropdownMenuItem key={team[0]} onClick={() => {
                                                                setCreateData((prev) => {
                                                                    return { ...prev, team_ids: [...prev.team_ids, team[0]]}
                                                                })
                                                            }}>{team[0] + " - " + team[1]}</DropdownMenuItem>
                                                        ))
                                                    }
                                                </DropdownMenuContent>
                                            </DropdownMenu>
                                            <ul className='flex flex-row gap-1 flex-wrap'>
                                                {teams.filter((team) => createData.team_ids.includes(team[0])).map((team) => (
                                                    <li key={team[0]}>
                                                        <button onClick={
                                                            () => {
                                                                setCreateData((prev) => {
                                                                    return { ...prev, team_ids: prev.team_ids.filter((team_id) => team_id !== team[0])}
                                                                })
                                                            }
                                                        }>
                                                            <div className="rounded-full border px-2">
                                                                {team[0] + " - " + team[1]}
                                                            </div>
                                                        </button>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                        <div className="w-full">
                                            <DropdownMenu>
                                                <DropdownMenuTrigger className="w-full">
                                                    <Button className="w-full" variant="outline">Positions</Button>
                                                </DropdownMenuTrigger>
                                                <DropdownMenuContent>
                                                    {
                                                        positions.map((position) => (
                                                            <DropdownMenuItem key={position[0]} onClick={() => {
                                                                setCreateData((prev) => {
                                                                    return { ...prev, position_ids: [...prev.position_ids, position[0]]}
                                                                })
                                                            }}>{position[1]}</DropdownMenuItem>
                                                        ))
                                                    }
                                                </DropdownMenuContent>
                                            </DropdownMenu>
                                            <ul className='flex flex-row gap-1 flex-wrap'>
                                                {positions.filter((position) => createData.position_ids.includes(position[0])).map((position) => (
                                                    <li key={position[0]}>
                                                        <button onClick={
                                                            () => {
                                                                setCreateData((prev) => {
                                                                    return { ...prev, position_ids: prev.position_ids.filter((position_id) => position_id !== position[0])}
                                                                })
                                                            }
                                                        }>
                                                            <div className="rounded-full border px-2">
                                                                {position[1]}
                                                            </div>
                                                        </button>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    </div>
                                </>
                            )
                        }
                        <Button className="w-full bg-zinc-900" onClick={
                            () => {
                                axios.post(`http://localhost:8000/api/create-user/`, {...createData, usertype: activeTab})
                                .then(function (response) {
                                    setCreateResponseView({status: "success", message: "User created successfully!"});
                                })
                                .catch(function (error) {
                                    console.log(error);
                                    setCreateResponseView({status: "error", message: "An error occured while creating user!"});
                                });
                            }
                        }>Create</Button>
                        <div className={createResponseView.status === "" ? "hidden" : (createResponseView.status === "success" ? "text-green-500" : "text-red-500")}>
                            {createResponseView.message}
                        </div>
                    </div>
                    <div className="flex-[30%] rounded-md shadow-sm border p-5 flex flex-col gap-2 h-min">
                        <h1 className="text-2xl font-bold">Update Stadium</h1>
                        <div className="flex flex-col gap-2 w-full">
                            <Select onValueChange={
                                (value) => {
                                    setUpdateData((prev) => {return { ...prev, previous_id: value}})
                                }
                            
                            }>
                                <SelectTrigger className="w-full">
                                    <SelectValue placeholder="Stadium" />
                                </SelectTrigger>
                                <SelectContent>
                                    {
                                        stadiums.map((stadium) => (
                                            <SelectItem key={stadium[2]} value={stadium[2]}>{stadium[0]}</SelectItem>
                                        ))
                                    }
                                </SelectContent>
                            </Select>
                            <Input placeholder="New Name" value={updateData.name} onChange={(e) => setUpdateData((prev) => {return { ...prev, name: e.target.value}})}/>
                            <Button className="w-full bg-zinc-900" onClick={
                                () => {
                                    axios.post(`http://localhost:8000/api/update-stadium/`, {previous_id: updateData.previous_id, name: updateData.name})
                                    .then(function (response) {
                                        setUpdateResponseView({status: "success", message: "Stadium updated successfully!"});
                                    })
                                    .catch(function (error) {
                                        console.log(error);
                                        setUpdateResponseView({status: "error", message: "An error occured while updating stadium!"});
                                    });  
                                }
                            }>Update</Button>
                        </div>
                        <div className={updateResponseView.status === "" ? "hidden" : (updateResponseView.status === "success" ? "text-green-500" : "text-red-500")}>
                            {updateResponseView.message}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
    )
}