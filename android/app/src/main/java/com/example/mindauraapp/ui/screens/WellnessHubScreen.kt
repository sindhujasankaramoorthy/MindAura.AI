package com.example.mindauraapp.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.mindauraapp.ui.components.GlassCard

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun WellnessHubScreen(navController: NavController) {
    Scaffold(
        topBar = { TopAppBar(title = { Text("Wellness Hub") }) }
    ) { padding ->
        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            contentPadding = PaddingValues(16.dp),
            modifier = Modifier.padding(padding).fillMaxSize(),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            item {
                GlassCard { Text("Meditation", modifier = Modifier.padding(16.dp)) }
            }
            item {
                GlassCard { Text("Breathing", modifier = Modifier.padding(16.dp)) }
            }
            item {
                GlassCard { Text("Yoga", modifier = Modifier.padding(16.dp)) }
            }
            item {
                GlassCard { Text("Sleep Stories", modifier = Modifier.padding(16.dp)) }
            }
        }
    }
}
