package edu.neu.mgen.finalproject5100.config;

// config/OpenAIConfig.java

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

// import okhttp3.OkHttpClient;
// import okhttp3.Request;
// import org.springframework.beans.factory.annotation.Value;
// import org.springframework.context.annotation.Bean;
// import org.springframework.context.annotation.Configuration;
// import retrofit2.Retrofit;
// import retrofit2.converter.gson.GsonConverterFactory;

// @Configuration
// public class OpenAIConfig {

//     @Value("${openai.api.key}")
//     private String openAiApiKey;

//     @Bean
//     public Retrofit openAiRetrofit() {
//         OkHttpClient client = new OkHttpClient.Builder()
//             .addInterceptor(chain -> {
//                 Request request = chain.request().newBuilder()
//                     .addHeader("Authorization", "Bearer " + openAiApiKey)
//                     .build();
//                 return chain.proceed(request);
//             }).build();

//         return new Retrofit.Builder()
//             .baseUrl("https://api.openai.com/")
//             .client(client)
//             .addConverterFactory(GsonConverterFactory.create())
//             .build();
//     }
// }

// @Configuration
// public class OpenAIConfig {
//     @Bean
//     public RestTemplate restTemplate() {
//         return new RestTemplate();
//     }
// }
