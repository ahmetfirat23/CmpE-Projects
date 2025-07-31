import { isEmpty } from 'lodash-es'

function getAuthUser() {
  const user = window.localStorage.getItem('userInfo')

  if (!user) return {}

  return JSON.parse(user)
}

const actions = {
  login: (user) => {
    window.localStorage.setItem('userInfo', JSON.stringify(user))
  },
  logout: () => {
    window.localStorage.removeItem('userInfo')
  },
  checkAuth: () => {
    const authUser = getAuthUser()

    if (!authUser || isEmpty(authUser)) {
      actions.logout()
      return false
    } else {
        return true
    }
  },
  getAuth: () => {
    return getAuthUser()
  }
}

function useAuth() {
  return {
    ...actions,
  }
}

export default useAuth