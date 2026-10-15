- Should we allow the model to write the ids for the containers' elements?





- We should allow the coordinates to all move in the same direction.
- The bounding boxes might be a bit different, should we compare to where the candidate best matched?
- Check the informaiton about the candidates that we are sending to the element mapper.
- Do you think our criterion to select candidates is weak? For instance, a container that happens to be in the same place and have teh same wxh ratio is considered a candidate, which is risky, as the element mapper might see similar elements and consider it a match. This would lead to coppying transitions states from one container to another wrongfully, just because we assumed they were variants. In spite of this, I think this is highly improbable to happen, so our solution might be good.