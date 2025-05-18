package com.example.violencedetector

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.widget.TextView

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val statusText: TextView = findViewById(R.id.statusText)
        statusText.text = "Violence Detector Initialized"
    }
}
