import logging


class DiscriminatorTrainer(object):
    threshold = 0.7  # 若機率>threshold，則規於1類，否則為0類

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def pretrain_D(self, D, G, train, criterion, optimizer, num_epoch=1,
                   src_field_name=None, tgt_field_name=None):
        src_name = self.src_field_name if src_field_name is None else src_field_name
        tgt_name = self.tgt_field_name if tgt_field_name is None else tgt_field_name

        for epoch in range(num_epoch):
            D.train()

            train_iter, = torchtext.data.BucketIterator.splits(
                (train,), batch_sizes=(1,), device=self.device,
                sort_key=lambda x: len(x.real_a), repeat=False)

            pool = helper.DDataPool(max_len=G.max_len)
            for batch in train_iter:
                src, src_length = getattr(batch, src_name)
                tgt, tgt_length = getattr(batch, tgt_name)
                rollout, _, _, _ = G.rollout(src, num_rollout=1)
                pool.append_fake(rollout[0, :].contiguous().view(1, -1))
                pool.append_real(G._validate_variables(target=tgt).data)

                if len(pool.fakes) > 1000:
                    break
            self.supervised_train_D(D, pool.batch_gen(), criterion, optimizer)

    def train(self, model, train_iter, crit, optimizer,
              src_field_name='seq', tgt_field_name='label'):
        model.train()

        total_loss, n_corrects, n_sample = 0, 0, 0
        for batch in train_iter:
            seq = getattr(batch, src_field_name)
            label = getattr(batch, tgt_field_name)
            batch_size = seq.size(0)

            optimizer.zero_grad()
            prob = model(seq)

            loss = crit(prob, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]
            n_corrects += (prob.gt(self.threshold).float().view(label.size()).data == label.data).sum()
            n_sample += batch_size

            # print(seq)
            # print(label)
            # print(prob)
            # print((prob.gt(self.threshold).float().view(batch_size).data == label.data).sum())

        acc = 100.0 * n_corrects / n_sample
        self.logger.info('(Train) - loss %.6f, acc %.4f%%(%d/%d)' % (
            total_loss, acc, n_corrects, n_sample))

    def evaluate(self, model, val_iter, crit,
                 src_field_name='seq', tgt_field_name='label'):
        model.eval()
        total_loss, n_corrects, n_sample = 0, 0, 0

        for batch in val_iter:
            seq = getattr(batch, src_field_name)
            label = getattr(batch, tgt_field_name)
            batch_size = seq.size(0)

            prob = model(seq)
            loss = crit(prob, label)

            total_loss += loss.data[0]
            n_corrects += (prob.gt(self.threshold).float().view(label.size()).data == label.data).sum()
            n_sample += batch_size

        acc = 100.0 * n_corrects / n_sample
        self.logger.info('(Eval) - loss %.6f, acc %.4f%%(%d/%d)' % (
            total_loss, acc, n_corrects, n_sample))
