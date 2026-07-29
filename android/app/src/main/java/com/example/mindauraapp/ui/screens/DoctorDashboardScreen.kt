package com.example.mindauraapp.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.mindauraapp.ui.components.GlassCard

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DoctorDashboardScreen(navController: NavController) {
    Scaffold(
        topBar = { TopAppBar(title = { Text("Doctor Dashboard") }) }
    ) { padding ->
        Column(modifier = Modifier.padding(padding).padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
            GlassCard(modifier = Modifier.fillMaxWidth()) {
                Text("Patient List", style = MaterialTheme.typography.titleMedium)
            }
            GlassCard(modifier = Modifier.fillMaxWidth()) {
                Text("Recent Alerts", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.error)
            }
            GlassCard(modifier = Modifier.fillMaxWidth()) {
                Text("Appointments", style = MaterialTheme.typography.titleMedium)
            }
        }
    }
}
