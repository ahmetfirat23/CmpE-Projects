import React, { useState, useEffect } from 'react';
import { Navigate, useNavigate } from 'react-router-dom'
import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import { z } from "zod"
import axios from 'axios';
import { useAuth } from '../hooks'

import { Button } from "@/components/ui/button"
import {
    Form,
    FormControl,
    FormDescription,
    FormField,
    FormItem,
    FormLabel,
    FormMessage,
} from "@/components/ui/form"
import { Input } from "@/components/ui/input"

const loginFormSchema = z.object({
    username: z.string(),
    password: z.string(),
  })


export default function Home() {
    useEffect(() => {
        axios.get(`http://localhost:8000/api/get-players/`)
            .then(function (response) {
                console.log(response.data.players);
            })
            .catch(function (error) {
                console.log(error);
            });
    }, [])

    const { checkAuth, login, getAuth } = useAuth()
    const isAuth = checkAuth()

    const navigate = useNavigate()

    if (isAuth) {
        const user = getAuth()
        if (user.type == "DBManager")
            return <Navigate to="/dbmanager" />
        else if (user.type == "Coach")
            return <Navigate to="/coach" />
        else if (user.type == "Jury")
            return <Navigate to="/jury" />
        else
            return <Navigate to="/player" />
    }

    const loginForm = useForm<z.infer<typeof loginFormSchema>>({
        resolver: zodResolver(loginFormSchema),
        defaultValues: {
            username: "",
            password: "",
        },
    })

    function onLoginSubmit(values: z.infer<typeof loginFormSchema>) {
        axios.post(`http://localhost:8000/api/login/`, values)
          .then(function (response) {
            loginForm.reset();
            login(response.data);
            if (response.data.type == "DBManager")
                navigate('/dbManager');
            else if (response.data.type == "Coach")
                navigate('/coach');
            else if (response.data.type == "Jury")
                navigate('/jury');
            else
                navigate('/player');
            
          })
          .catch(function (error) {
            console.log(error);
            setLoginResponseView({
                status: "error",
                message: "Invalid username or password"
            })
          });        
    }

    const [loginResponseView, setLoginResponseView] = useState<string>({
        status: "",
        message: ""
    })

    return (
        <div className="flex justify-center w-screen h-screen">
            <div className='flex flex-col w-full h-full justify-center items-center'>
                <div className="flex flex-col w-1/4 items-center gap-5 rounded-md shadow-sm border p-7">
                    <div className="text-2xl font-bold">
                        Login
                    </div>
                    <div className="flex flex-col gap-2 w-full">
                    <Form {...loginForm}>
                            <form onSubmit={loginForm.handleSubmit(onLoginSubmit)} className="space-y-2">
                                <FormField
                                control={loginForm.control}
                                name="username"
                                render={({ field }) => (
                                    <FormItem>
                                    <FormControl>
                                        <Input placeholder="Username" {...field} />
                                    </FormControl>
                                    <FormMessage />
                                    </FormItem>
                                    
                                )}
                                />
                                <FormField
                                control={loginForm.control}
                                name="password"
                                render={({ field }) => (
                                    <FormItem>
                                    <FormControl>
                                        <Input placeholder="Password" type='password' {...field} />
                                    </FormControl>
                                    <FormMessage />
                                    </FormItem>
                                    
                                )}
                                />
                                <Button className="w-full bg-zinc-900" type="submit">Continue</Button>
                            </form>
                        </Form>
                        <div className={loginResponseView.status === "" ? "hidden" : (loginResponseView.status === "success" ? "text-green-500" : "text-red-500")}>
                            {loginResponseView.message}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
