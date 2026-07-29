package com.example.mindauraapp.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.mindauraapp.ui.components.GlassCard
import com.example.mindauraapp.ui.components.GradientButton

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AiAnalysisScreen(navController: NavController) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("AI Analysis") },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            item {
                GlassCard(modifier = Modifier.fillMaxWidth()) {
                    Column {
                        Text("Primary Emotion: Frustration", style = MaterialTheme.typography.titleMedium)
                        Text("Secondary Emotion: Anger", style = MaterialTheme.typography.titleMedium)
                        Text("Confidence: 85%", style = MaterialTheme.typography.bodyLarge)
                    }
                }
            }
            item {
                GlassCard(modifier = Modifier.fillMaxWidth()) {
                    Column {
                        Text("Risk Assessment", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                        Spacer(modifier = Modifier.height(8.dp))
                        Text("Stress Level: High", color = MaterialTheme.colorScheme.error)
                        Text("Depression Risk: Low")
                        Text("Anxiety Risk: Moderate")
                        Text("Burnout Risk: High")
                    }
                }
            }
            item {
                Text("AI Explanation", style = MaterialTheme.typography.titleMedium)
                Text("Vocal characteristics align with Frustration. The speaker expresses 'disappointment' in language, but vocal characteristics indicate 'Frustration'.", style = MaterialTheme.typography.bodyMedium)
            }
            item {
                Spacer(modifier = Modifier.height(16.dp))
                GradientButton(text = "Breathing Exercise", onClick = { /* TODO */ })
                Spacer(modifier = Modifier.height(8.dp))
                OutlinedButton(onClick = { /* TODO */ }, modifier = Modifier.fillMaxWidth()) {
                    Text("Meditation")
                }
            }
        }
    }
}
