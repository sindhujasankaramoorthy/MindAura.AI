package com.example.mindauraapp.ui.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.mindauraapp.ui.screens.*

@Composable
fun MainNavigation() {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = Screen.Splash.route) {
        composable(Screen.Splash.route) {
            SplashScreen(onNavigateToLogin = {
                navController.navigate(Screen.Login.route) {
                    popUpTo(Screen.Splash.route) { inclusive = true }
                }
            })
        }
        composable(Screen.Login.route) {
            LoginScreen(onLoginSuccess = {
                navController.navigate(Screen.Dashboard.route) {
                    popUpTo(Screen.Login.route) { inclusive = true }
                }
            })
        }
        composable(Screen.Dashboard.route) {
            DashboardScreen(navController = navController)
        }
        composable(Screen.Journal.route) {
            JournalScreen(navController = navController)
        }
        composable(Screen.AiAnalysis.route) {
            AiAnalysisScreen(navController = navController)
        }
        composable(Screen.EmotionHistory.route) {
            EmotionHistoryScreen(navController = navController)
        }
        composable(Screen.Community.route) {
            CommunityScreen(navController = navController)
        }
        composable(Screen.EmergencySupport.route) {
            EmergencySupportScreen(navController = navController)
        }
        composable(Screen.ContactsManagement.route) {
            ContactsManagementScreen(navController = navController)
        }
        composable(Screen.DoctorDashboard.route) {
            DoctorDashboardScreen(navController = navController)
        }
        composable(Screen.WellnessHub.route) {
            WellnessHubScreen(navController = navController)
        }
        composable(Screen.Profile.route) {
            ProfileScreen(navController = navController)
        }
    }
}
