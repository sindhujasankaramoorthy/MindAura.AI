package com.example.mindauraapp.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun JournalScreen(navController: NavController) {
    var journalText by remember { mutableStateOf("") }
    var emotionLevel by remember { mutableStateOf(5f) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Journal") },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
        ) {
            OutlinedTextField(
                value = journalText,
                onValueChange = { journalText = it },
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                placeholder = { Text("How are you feeling today?") }
            )
            Spacer(modifier = Modifier.height(16.dp))
            
            Text("Emotion Level: ${emotionLevel.toInt()}")
            Slider(
                value = emotionLevel,
                onValueChange = { emotionLevel = it },
                valueRange = 0f..10f,
                steps = 10
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                OutlinedButton(onClick = { /* Voice Journal */ }) {
                    Icon(Icons.Default.Mic, contentDescription = "Voice Journal")
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Voice Record")
                }
                Button(onClick = { /* Analyze */ }) {
                    Text("AI Analysis")
                }
            }
        }
    }
}
