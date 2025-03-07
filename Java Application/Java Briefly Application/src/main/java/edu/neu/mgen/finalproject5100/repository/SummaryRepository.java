package edu.neu.mgen.finalproject5100.repository;

import edu.neu.mgen.finalproject5100.model.Summary;
import edu.neu.mgen.finalproject5100.util.DateUtil;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.repository.query.Param;

import java.util.Date;
import java.util.Optional;
// import org.springframework.data.jpa.repository.JpaRepository;
// import org.springframework.data.jpa.repository.Query;
// import java.util.Optional;

import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.*;
import com.google.firebase.cloud.FirestoreClient;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutionException;

@Repository
public class SummaryRepository {
    private static final String COLLECTION_NAME = "summaries";

    public Summary save(Summary summary, String userId) {
        try {
            Firestore firestore = FirestoreClient.getFirestore();

            // If no ID exists, generate one
            if (summary.getId() == null) {
                summary.setId(firestore.collection(COLLECTION_NAME).document().getId());
            }

            // Assign the userId to the summary
            summary.setUserId(userId);

            ApiFuture<WriteResult> future = firestore
                    .collection(COLLECTION_NAME)
                    .document(summary.getId())
                    .set(summary);
            future.get(); // Wait for the operation to complete
            return summary;
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error saving summary", e);
        }
    }


    public Summary saveOrUpdate(Summary summary, String userId) {
        try {
            Firestore firestore = FirestoreClient.getFirestore();
            CollectionReference summaries = firestore.collection(COLLECTION_NAME);

            // Query for an existing summary by userId and articleId
            ApiFuture<QuerySnapshot> query = summaries
                    .whereEqualTo("userId", summary.getUserId())
                    .whereEqualTo("articleId", summary.getArticleId())
                    .limit(1)
                    .get();

            List<QueryDocumentSnapshot> documents = query.get().getDocuments();

            if (!documents.isEmpty()) {
                // Update existing document
                DocumentReference docRef = documents.get(0).getReference();
                docRef.update("score", summary.getScore());
                docRef.update("feedback", summary.getFeedback());
                docRef.update("userSummary", summary.getUserSummary());
                docRef.update("submissionDate", new Date());
                return summary;
            } else {
                // Save a new summary if no match is found
                if (summary.getId() == null) {
                    summary.setId(summaries.document().getId());
                }
                summaries.document(summary.getId()).set(summary).get();
                return summary;
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error saving or updating summary", e);
        }
    }

    public int countSubmissions(String userId, LocalDate sinceDate) {
        System.out.println("userId: " + userId);
        System.out.println("sinceDate: " + sinceDate);
        try {
            Firestore firestore = FirestoreClient.getFirestore();
            ApiFuture<QuerySnapshot> future = firestore.collection(COLLECTION_NAME)
                    .whereEqualTo("userId", userId)
                    .whereGreaterThanOrEqualTo("submissionDate", Date.from(sinceDate.atStartOfDay(ZoneId.systemDefault()).toInstant()))
                    .get();
            return future.get().size();
        } catch (InterruptedException | ExecutionException e) {
            System.out.println("Error: " + e.getMessage());
            throw new RuntimeException("Error counting submissions", e);
        }
    }

    public List<Summary> getSubmissions(String userId, int page, int pageSize) {
        try {
            Firestore firestore = FirestoreClient.getFirestore();
            ApiFuture<QuerySnapshot> future = firestore.collection(COLLECTION_NAME)
                    .whereEqualTo("userId", userId)
                    .orderBy("submissionDate", Query.Direction.DESCENDING)
                    .offset((page - 1) * pageSize)
                    .limit(pageSize)
                    .get();

            List<Summary> submissions = new ArrayList<>();
            for (QueryDocumentSnapshot document : future.get().getDocuments()) {
                submissions.add(document.toObject(Summary.class));
            }
            return submissions;
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error retrieving submissions", e);
        }
    }

      public List<Summary> getFeedback(String userId, Date submissionDate) {
        try {
            Firestore firestore = FirestoreClient.getFirestore();
            CollectionReference collection = firestore.collection(COLLECTION_NAME);

            Query query = collection.whereEqualTo("userId", userId);
            if (submissionDate != null) {

                Date start_date = DateUtil.getStartOfDate(submissionDate);
                Date end_date = DateUtil.getEndOfDate(submissionDate);
                query = query.whereGreaterThanOrEqualTo("submissionDate", start_date)
                        .whereLessThanOrEqualTo("submissionDate", end_date);
                //query = query.whereEqualTo("submissionDate", submissionDate);
            }

            ApiFuture<QuerySnapshot> querySnapshot = query.get();
            List<Summary> feedbackList = new ArrayList<>();
            for (QueryDocumentSnapshot document : querySnapshot.get().getDocuments()) {
                feedbackList.add(document.toObject(Summary.class));
            }
            return feedbackList;
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error retrieving feedback", e);
        }
    }

    public List<String> getFeedbackDates(String userId, Date submissionDate) {
        try {
            Firestore firestore = FirestoreClient.getFirestore();
            CollectionReference collection = firestore.collection(COLLECTION_NAME);
    
            Query query = collection.whereEqualTo("userId", userId);
    
            if (submissionDate != null) {
                // Get the start and end of the day for the submissionDate
                Date start_date = DateUtil.getStartOfDate(submissionDate);
                Date end_date = DateUtil.getEndOfDate(submissionDate);
    
                query = query.whereGreaterThanOrEqualTo("submissionDate", start_date)
                             .whereLessThanOrEqualTo("submissionDate", end_date);
            }
    
            ApiFuture<QuerySnapshot> querySnapshot = query.get();
            List<String> feedbackDates = new ArrayList<>();
    
            // Process each document in the query result
            for (QueryDocumentSnapshot document : querySnapshot.get().getDocuments()) {
                Summary summary = document.toObject(Summary.class);
                Date feedbackDate = summary.getSubmissionDate();  // Assuming your Summary class has submissionDate field
    
                // Extract the year, month, and day from the feedbackDate
                Calendar calendar = Calendar.getInstance();
                calendar.setTime(feedbackDate);
                int year = calendar.get(Calendar.YEAR);
                int month = calendar.get(Calendar.MONTH) + 1; // Adjust for 0-based month
                int dayOfMonth = calendar.get(Calendar.DAY_OF_MONTH);
    
                // Format the date as "YYYY-MM-DD"
                String formattedDate = String.format("%d-%02d-%02d", year, month, dayOfMonth);
                
                // Add the formatted date to the list if it's not already there
                if (!feedbackDates.contains(formattedDate)) {
                    feedbackDates.add(formattedDate);
                }
            }
    
            return feedbackDates;
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error retrieving feedback dates", e);
        }
    }
    



    public Summary findBestScoreByArticleId(String articleId, String userId) {
        try {
            Firestore firestore = FirestoreClient.getFirestore();

            // Query for summaries with the given articleId and userId, ordered by score
            ApiFuture<QuerySnapshot> future = firestore
                    .collection(COLLECTION_NAME)
                    .whereEqualTo("articleId", articleId)
                    .whereEqualTo("userId", userId)
                    // .orderBy("score", Query.Direction.DESCENDING)
                    .limit(1)
                    .get();

            List<QueryDocumentSnapshot> documents = future.get().getDocuments();
            if (!documents.isEmpty()) {
                return documents.get(0).toObject(Summary.class);
            }
            return null;
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error finding best score", e);
        }
    }

    public List<Summary> findAllByArticleId(String articleId, String userId) {
        try {
            Firestore firestore = FirestoreClient.getFirestore();

            ApiFuture<QuerySnapshot> future = firestore
                    .collection(COLLECTION_NAME)
                    .whereEqualTo("articleId", articleId)
                    .whereEqualTo("userId", userId)
                    // .orderBy("submissionDate", Query.Direction.DESCENDING)
                    .get();

            List<Summary> summaries = new ArrayList<>();
            List<QueryDocumentSnapshot> documents = future.get().getDocuments();
            for (QueryDocumentSnapshot document : documents) {
                summaries.add(document.toObject(Summary.class));
            }
            return summaries;
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Error finding summaries", e);
        }
    }


    public List<Summary> findAllByUserId(String userId, Date submissionDateLessThan, int limit) {
        try {
            if (submissionDateLessThan == null) {
                submissionDateLessThan = new Date();
            }
            if (userId == null) {
                throw new IllegalArgumentException("userId cannot be null");
            }
            if (limit <= 0) {
                throw new IllegalArgumentException("limit must be greater than 0");
            }

            Firestore firestore = FirestoreClient.getFirestore();

            ApiFuture<QuerySnapshot> future = firestore
                    .collection(COLLECTION_NAME)
                    .whereLessThan("submissionDate", submissionDateLessThan)
                    .whereEqualTo("userId", userId)
                    .orderBy("submissionDate", Query.Direction.DESCENDING)
                    .limit(limit)
                    .get();

            List<Summary> summaries = new ArrayList<>();
            List<QueryDocumentSnapshot> documents = future.get().getDocuments();
            for (QueryDocumentSnapshot document : documents) {
                summaries.add(document.toObject(Summary.class));
            }
            return summaries;
        } catch (Exception e) {
            throw new RuntimeException("Error finding summaries", e);
        }
    }

}
