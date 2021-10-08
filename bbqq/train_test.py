import torch
from tqdm.autonotebook import tqdm

from bbqq.f1_score import f1_loss, f1score, calc_accuracy
from torch.nn import functional as F


def train_test(EPOCHS=None, model=None, train_dataloader=None, test_dataloader=None, optimizer=None, log_interval=None,
          max_grad_norm=None, scheduler=None):
    for e_idx, epoch in enumerate(range(EPOCHS)):
        losses = list()
        test_f1 = 0.0

        model.train()
        for b_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            X, y = batch
            loss = model.training_step(X, y)

            loss.backward()  # backprop the loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()  # gradient step
            scheduler.step()  # resetting the gradients.

            y_hat = model.predict(X)
            y_hat = F.softmax(y_hat, dim=1)
            # train_f1 = f1score(y, y_hat)
            train_f1 = calc_accuracy(y_hat, y)  # accuracy
            losses.append(loss.item())
            avg_loss = (sum(losses) / len(losses))
            print("epoch {} batch id {} loss {} train acc {}".format(e_idx+1, b_idx + 1, loss.item(), train_f1))
        print("epoch {} avg_loss {} train acc {} ".format(e_idx+1, avg_loss, train_f1))

        model.eval()
        for b_idx, batch in enumerate(tqdm(test_dataloader)):
            X, y = batch
            y_hat = model.predict(X)
            y_hat = F.softmax(y_hat, dim=1)
            test_f1 += calc_accuracy(y_hat, y)  # accuracy
        print(f"epoch {e_idx+1} test acc {test_f1}" )

    torch.save(model, 'model.pth')


def train_test_2(EPOCHS=None, model=None, train_dataloader=None, test_dataloader=None, optimizer=None):
    for epoch in range(EPOCHS):
        losses = list()
        test_f1 = 0.0
        model.train()
        for b_idx, batch in enumerate(train_dataloader):
            X, y = batch
            loss = model.training_step(X, y)
            loss.backward()  # 오차 역전파
            optimizer.step()  # 경사도 하강
            optimizer.zero_grad()  # 기울기 축적방지
            print(f"epoch:{epoch+1}, loss:{loss.item()} ")

        model.eval()
        for b_idx, batch in enumerate(tqdm(test_dataloader)):
            X, y = batch
            y_hat = model.predict(X)
            y_hat = F.softmax(y_hat, dim=1)
            test_f1 += calc_accuracy(y_hat, y)  # accuracy
        print(f"epoch {epoch+1} test acc {test_f1}" )