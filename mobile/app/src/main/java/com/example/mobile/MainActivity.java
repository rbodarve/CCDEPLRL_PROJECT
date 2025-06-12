package com.example.mobile;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;


public class MainActivity extends AppCompatActivity {
    Button gru;
    Button base;
    Button tsm;
    Button conv;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_options);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
        gru = findViewById(R.id.gru);
        base = findViewById(R.id.base);
        tsm = findViewById(R.id.tsm);
        conv = findViewById(R.id.conv);
        gru.setOnClickListener(v -> startIntent_with_key("gru"));
        base.setOnClickListener(v -> startIntent_with_key("base"));
        tsm.setOnClickListener(v -> startIntent_with_key("tsm"));
        conv.setOnClickListener(v -> startIntent_with_key("conv"));
    }

    public void startIntent_with_key(String modelVal){
        Intent intent = new Intent(MainActivity.this, TFCamera.class);
        intent.putExtra("model", modelVal);
        startActivity(intent);
    }

}