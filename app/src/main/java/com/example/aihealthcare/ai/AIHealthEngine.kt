package com.example.aihealthcare.ai

import android.util.Log
import com.example.aihealthcare.BuildConfig
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.POST

// Data models for Groq (OpenAI-compatible)
data class GroqRequest(
    val model: String,
    val messages: List<GroqMessage>,
    val temperature: Double = 0.7
)

data class GroqMessage(
    val role: String,
    val content: String
)

data class GroqResponse(
    val choices: List<GroqChoice>
)

data class GroqChoice(
    val message: GroqMessage
)

interface GroqApiService {
    @POST("v1/chat/completions")
    suspend fun getChatCompletion(@Body request: GroqRequest): GroqResponse
}

object AIHealthEngine {
    private const val TAG = "AIHealthEngine"
    // API Key is loaded from BuildConfig which reads from local.properties
    // DO NOT hardcode API keys - they will be caught by GitHub secret scanning
    private val API_KEY = BuildConfig.GROQ_API_KEY
    private const val MODEL_NAME = "llama-3.3-70b-versatile"

    private val client = OkHttpClient.Builder()
        .addInterceptor { chain ->
            val request = chain.request().newBuilder()
                .addHeader("Authorization", "Bearer $API_KEY")
                .addHeader("Content-Type", "application/json")
                .build()
            chain.proceed(request)
        }
        .addInterceptor(HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BODY
        })
        .build()

    private val groqApi = Retrofit.Builder()
        .baseUrl("https://api.groq.com/openai/")
        .client(client)
        .addConverterFactory(GsonConverterFactory.create())
        .build()
        .create(GroqApiService::class.java)

    suspend fun getAIResponse(prompt: String, userProfile: String = ""): String {
        Log.d(TAG, "Requesting Groq AI for prompt: $prompt")
        
        val systemMessage = """
            Act as a highly professional and empathetic health advisor. 
            User Context: $userProfile
            Guidelines:
            - Provide evidence-based, safe, and actionable health advice.
            - NEVER diagnose diseases. 
            - Always advise consulting a doctor if the situation sounds serious.
            - Keep responses structured with headings and bullet points where appropriate.
            - Avoid raw markdown symbols like ** in plain text, use professional formatting.
            - If user has chronic conditions (like Diabetes/BP) mentioned in context, factor that into safety advice.
        """.trimIndent()
        
        return try {
            val request = GroqRequest(
                model = MODEL_NAME,
                messages = listOf(
                    GroqMessage("system", systemMessage),
                    GroqMessage("user", prompt)
                )
            )
            
            val response = groqApi.getChatCompletion(request)
            val text = response.choices.firstOrNull()?.message?.content
            
            if (text != null) {
                Log.d(TAG, "Success: Response received from Groq")
                text
            } else {
                Log.w(TAG, "Warning: Groq response text was null")
                "I'm sorry, I couldn't process that. Please try again."
            }
        } catch (e: Exception) {
            val errorMsg = e.localizedMessage ?: "Groq API error"
            Log.e(TAG, "Groq Error: $errorMsg", e)
            
            when {
                errorMsg.contains("401") -> "Error: Invalid Groq API Key."
                errorMsg.contains("429") -> "Error: Groq rate limit reached. Please wait."
                errorMsg.contains("Unable to resolve host") -> "Error: No internet connection."
                else -> "Error: $errorMsg"
            }
        }
    }
}
