package com.example.mindauraapp.ui.navigation

sealed class Screen(val route: String) {
    object Splash : Screen("splash")
    object Login : Screen("login")
    object Dashboard : Screen("dashboard")
    object Journal : Screen("journal")
    object AiAnalysis : Screen("ai_analysis")
    object EmotionHistory : Screen("emotion_history")
    object Community : Screen("community")
    object EmergencySupport : Screen("emergency_support")
    object ContactsManagement : Screen("contacts_management")
    object DoctorDashboard : Screen("doctor_dashboard")
    object WellnessHub : Screen("wellness_hub")
    object Profile : Screen("profile")
}
