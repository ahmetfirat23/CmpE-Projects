import { useNavigate, Navigate} from 'react-router-dom'
import { useAuth } from '../hooks'
import LogoutOutlinedIcon from '@mui/icons-material/LogoutOutlined';
import React, { useState, useEffect } from 'react';
import axios from 'axios';

import {Input} from '@/components/ui/input'
import { Button } from "@/components/ui/button"
import { set } from 'date-fns';

import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
  } from "@/components/ui/select"

export default function Jury() {
    const navigate = useNavigate()
    const { logout, checkAuth, getAuth } = useAuth()
    const isAuth = checkAuth()

    const user = getAuth()

    const [sessions, setSessions] = useState([]);

    useEffect(() => {
        axios.post(`http://localhost:8000/api/get-jury-sessions/`, {jury_username: user.user[0]})
        .then(function (response) {
            setSessions(response.data.sessions);
        })
        .catch(function (error) {
            console.log(error);
        });
    }, []);


    if (!isAuth) return <Navigate to="/" />

    const [activeTab, setActiveTab] = React.useState('rate')

    const [statsTabState, setStatsTabState] = React.useState({averageRating: 0, ratingCount: 0})
    const [rateTabState, setRateTabState] = React.useState({sessionID: null, rating: null})

    const [rateResponseView, setRateResponseView] = React.useState({
        status: "",
        message: ""
    })
    const [statsResponseView, setStatsResponseView] = React.useState({
        status: "",
        message: ""
    })
    return (
        <div className='h-screen w-screen flex flex-col'>
            <div className='flex-initial flex flex-row justify-between bg-zinc-900 text-white py-3 px-5 items-center'>
                <h1 className='text-xl'>Jury</h1>
                <div className='flex flex-row bg-zinc-700 p-1 text-base gap-1 rounded-sm text-white w-3/12'>
                    <button className={activeTab === "rate" ? "bg-zinc-900 rounded-sm flex-1 p-1" : "flex-1 p-1 text-zinc-400"} onClick={() => setActiveTab('rate')}>Rate</button>
                    <button className={activeTab === "stats" ? "bg-zinc-900 rounded-sm flex-1 p-1" : "flex-1 p-1 text-zinc-400"} onClick={() => {
                        axios.post(`http://localhost:8000/api/view-rating-stats/`, {jury_username: user.user[0]})
                        .then(function (response) {
                            setStatsTabState({averageRating: response.data.average_rating, ratingCount: response.data.rating_count});
                        })
                        .catch(function (error) {
                            console.log(error);
                            setStatsResponseView({
                                status: "error",
                                message: "An error occurred while fetching stats!"
                            })
                        });
                        setActiveTab('stats');
                    }}>Stats</button>
                </div>
                <button onClick={() => {logout(); navigate('/')}}><LogoutOutlinedIcon className='text-2xl'/></button>
            </div>
            <div className='h-full w-full flex-auto flex justify-center'>
                <div className='flex flex-col justify-center'>
                    <div className='shadow-sm rounded-md border p-7'>
                        {
                            activeTab === 'stats' && (
                                <div>
                                    <h1 className="text-2xl font-bold">Stats</h1>
                                    <div>{`Average rating: ${statsTabState.averageRating}`}</div>
                                    <div>{`Rating count: ${statsTabState.ratingCount}`}</div>
                                    <div className={statsResponseView.status === "" ? "hidden" : "text-red-500"}>
                                        {statsResponseView.message}
                                    </div>
                                </div>
                            )
                        }
                        {
                            activeTab === 'rate' && (
                                <div className='flex flex-col gap-2'>
                                    <h1 className="text-2xl font-bold">Rate a Match Session</h1>
                                    <Select onValueChange={
                                        (value) => {
                                            setRateTabState((prev) => {return { ...prev, sessionID: value}})
                                        }
                                    
                                    }>
                                        <SelectTrigger className="w-full">
                                            <SelectValue placeholder="Session ID" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {
                                                sessions.map((session) => (
                                                    <SelectItem key={session[0]} value={session[0]}>{session[0]}</SelectItem>
                                                ))
                                            }
                                        </SelectContent>
                                    </Select>
                                    <Input placeholder='Rating' className='border' value={rateTabState.rating} onChange={(e) => setRateTabState((prev) => {return { ...prev, rating: e.target.value}})}/>
                                    <Button onClick={() => {
                                        axios.post(`http://localhost:8000/api/rate-match-session/`, {
                                            session_id: rateTabState.sessionID,
                                            rating: parseFloat(rateTabState.rating),
                                            jury_username: user.user[0]
                                        })
                                        .then(function (response) {
                                            setRateResponseView({
                                                status: "success",
                                                message: "Rating submitted successfully!"
                                            })
                                        })
                                        .catch(function (error) {
                                            console.log(error);
                                            setRateResponseView({
                                                status: "error",
                                                message: "An error occurred while submitting rating!"
                                            })
                                        });
                                    }}>Rate</Button>
                                    <div className={rateResponseView.status === "" ? "hidden" : (rateResponseView.status === "success" ? "text-green-500" : "text-red-500")}>
                                        {rateResponseView.message}
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