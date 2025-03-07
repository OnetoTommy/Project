package edu.neu.mgen.finalproject5100.controller;

import edu.neu.mgen.finalproject5100.service.NewsService;
import edu.neu.mgen.finalproject5100.util.DateUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import edu.neu.mgen.finalproject5100.model.Article;
import edu.neu.mgen.finalproject5100.repository.ArticleRepository;

import java.util.Date;

@RestController
@RequestMapping("/api/articles")
@CrossOrigin(origins = "*")
public class ArticleController {

    @Autowired
    private NewsService newsService;

    @Autowired
    ArticleRepository articleRepo;

    @GetMapping("/id/{id}")
    public ResponseEntity<?> getArticle(@PathVariable String id) {
        Article article = articleRepo.findById(id);
        return ResponseEntity.ok(article);
    }


    @GetMapping("/latest")
    public ResponseEntity<?> getLatest() {
        //updateNews();
        Article article = articleRepo.findLatest();
        return ResponseEntity.ok(article);
    }

    //todo: migrate this to schedule tasks
    private Article updateNews() {
        //check if we already have news
        Article articleInDb = articleRepo.findByDate(new Date());
        if (articleInDb != null) {
            return articleInDb;
        }

        //retrieve news
        Article articleFromAPI = newsService.getLatest();
        Date today = DateUtil.getStartOfDate(new Date());
        if (!today.equals(articleFromAPI.getDate())) {
            return null;
        }

        //save
        articleRepo.save(articleFromAPI);
        return articleFromAPI;
    }
}
