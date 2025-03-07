package edu.neu.mgen.finalproject5100.repository;

import java.util.*;

import edu.neu.mgen.finalproject5100.model.Article;
import edu.neu.mgen.finalproject5100.util.DateUtil;
import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.*;
import com.google.firebase.cloud.FirestoreClient;
import org.springframework.stereotype.Repository;

import java.util.concurrent.ExecutionException;

@Repository
public class ArticleRepository {
    private static final String COLLECTION_NAME = "articles";

    public Article save(Article article) {
        if (article == null) {
            return null;
        }
        try {
            Firestore firestore = FirestoreClient.getFirestore();

            // If no ID exists, generate one
            if (article.getId() == null) {
                article.setId(firestore.collection(COLLECTION_NAME).document().getId());
            }

            ApiFuture<WriteResult> future = firestore.collection(COLLECTION_NAME).document(article.getId()).set(article);

            future.get(); // Wait for the operation to complete
            return article;
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error saving article", e);
        }
    }

    public Article findById(String id) {
        if (id == null) {
            return null;
        }

        try {
            Firestore firestore = FirestoreClient.getFirestore();
            ApiFuture<QuerySnapshot> future = firestore.collection(COLLECTION_NAME).whereEqualTo("id", id).limit(1).get();

            List<QueryDocumentSnapshot> documents = future.get().getDocuments();
            if (!documents.isEmpty()) {
                return documents.getFirst().toObject(Article.class);
            }
            return null;
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error saving article", e);
        }
    }

    public ArrayList<Article> findAllByIds(ArrayList<String> ids) {
        if (ids == null || ids.isEmpty()) {
            return null;
        }
        if (ids.size() > 100) {
            throw new RuntimeException("findByIds only accept 100 ids");
        }

        try {
            Firestore firestore = FirestoreClient.getFirestore();
            ApiFuture<QuerySnapshot> future = firestore.collection(COLLECTION_NAME).whereIn("id", ids).limit(100).get();
            List<QueryDocumentSnapshot> documents = future.get().getDocuments();

            ArrayList<Article> articles = new ArrayList<>();
            if (documents.isEmpty()) {
                return null;
            }
            for (QueryDocumentSnapshot document : documents) {
                articles.add(document.toObject(Article.class));
            }
            return articles;
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error finding articles", e);
        }
    }

    public Article findByDate(Date date) {
        if (date == null) {
            return null;
        }

        Date dateStart = DateUtil.getStartOfDate(date);
        try {
            Firestore firestore = FirestoreClient.getFirestore();

            ApiFuture<QuerySnapshot> future = firestore.collection(COLLECTION_NAME).whereEqualTo("date", dateStart).limit(1).get();

            List<QueryDocumentSnapshot> documents = future.get().getDocuments();
            if (!documents.isEmpty()) {
                return documents.getFirst().toObject(Article.class);
            }
            return null;
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error finding article", e);
        }

    }

    public Article findLatest() {
        try {
            Firestore firestore = FirestoreClient.getFirestore();
            ApiFuture<QuerySnapshot> future = firestore.collection(COLLECTION_NAME).orderBy("date", Query.Direction.DESCENDING).limit(1).get();

            List<QueryDocumentSnapshot> documents = future.get().getDocuments();
            if (!documents.isEmpty()) {
                return documents.getFirst().toObject(Article.class);
            }
            return null;
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error finding article", e);
        }
    }

}
