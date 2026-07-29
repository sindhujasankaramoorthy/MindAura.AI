package com.example.mindauraapp.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.mindauraapp.ui.navigation.Screen

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EmergencySupportScreen(navController: NavController) {
    Scaffold(
        topBar = { TopAppBar(title = { Text("Emergency Support") }) }
    ) { padding ->
        Column(
            modifier = Modifier.padding(padding).padding(24.dp).fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Button(
                onClick = { /* Trigger SOS */ },
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
                modifier = Modifier.size(200.dp)
            ) {
                Text("SOS", style = MaterialTheme.typography.displayLarge, color = Color.White)
            }
            
            Spacer(modifier = Modifier.height(32.dp))
            
            OutlinedButton(
                onClick = { navController.navigate(Screen.ContactsManagement.route) },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Manage Emergency Contacts")
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            OutlinedButton(onClick = { /* Call Helpline */ }, modifier = Modifier.fillMaxWidth()) {
                Text("National Mental Health Helpline")
            }
            Spacer(modifier = Modifier.height(16.dp))
            OutlinedButton(onClick = { /* Share Location */ }, modifier = Modifier.fillMaxWidth()) {
                Text("Share Live Location")
            }
        }
    }
}
