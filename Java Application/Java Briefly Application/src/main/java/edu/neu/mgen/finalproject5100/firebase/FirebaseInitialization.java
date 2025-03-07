package edu.neu.mgen.finalproject5100.firebase;

import java.io.IOException;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ClassPathResource;

// "/Users/ebunadebesin/Desktop/finalproject5100/src/main/resources/serviceAccount.json"

@Configuration
public class FirebaseInitialization {
    @Bean
    public FirebaseApp firebaseApp() throws IOException {
        // Load the service account key JSON file from resources
        ClassPathResource serviceAccount = new ClassPathResource("serviceAccount.json");
        
        FirebaseOptions options = FirebaseOptions.builder()
            .setCredentials(GoogleCredentials.fromStream(serviceAccount.getInputStream()))
            .build();

        // Initialize the app if it hasn't been initialized yet
        if (FirebaseApp.getApps().isEmpty()) {
            return FirebaseApp.initializeApp(options);
        }
        return FirebaseApp.getInstance();
    }
}
