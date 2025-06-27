package com.example.myapplication;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private EditText editTextCommand;
    private Button buttonSend;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        editTextCommand = findViewById(R.id.editTextCommand);
        buttonSend = findViewById(R.id.buttonSend);

        buttonSend.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String command = editTextCommand.getText().toString().trim();
                if (!command.isEmpty()) {
                    // 向服务器发送指令
                    sendCommandToServer(command);
                } else {
                    Toast.makeText(MainActivity.this, "请输入指令", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    private void sendCommandToServer(final String command) {
        new Thread(new Runnable() {
            @Override
            public void run() {
                OkHttpClient client = new OkHttpClient();

                // 替换成你的服务器地址
                String serverUrl = getString(R.string.server_address);

                MediaType mediaType = MediaType.parse("text/plain; charset=utf-8");
                RequestBody requestBody = RequestBody.create(mediaType, command);
                Request request = new Request.Builder()
                        .url(serverUrl)
                        .post(requestBody)
                        .build();

                try {
                    Response response = client.newCall(request).execute();
                    if (response.isSuccessful()) {
                        final String responseData = response.body().string();
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                Toast.makeText(MainActivity.this, "指令发送成功：" + responseData, Toast.LENGTH_SHORT).show();
                            }
                        });
                    } else {
                        Log.e("MainActivity", response.toString());
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                Toast.makeText(MainActivity.this, "指令发送失败", Toast.LENGTH_SHORT).show();
                            }
                        });
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                    final String errorMessage = e.getMessage(); // 获取异常信息
                    Log.e("MainActivity", errorMessage);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(MainActivity.this, "网络错误: " + errorMessage, Toast.LENGTH_SHORT).show();
                        }
                    });
                }
            }
        }).start();
    }
}
